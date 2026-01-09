import os
import tarfile
import tempfile
import subprocess
import time
import shutil


class Solution:
    def _try_objdump_fuzz(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball
            extracted_root = tmpdir
            extracted_ok = False
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(path=tmpdir)
                extracted_ok = True
            except Exception:
                pass

            if not extracted_ok:
                try:
                    shutil.unpack_archive(src_path, tmpdir)
                    extracted_ok = True
                except Exception:
                    return None

            # Find configure script, preferring one that has a 'binutils' subdir
            config_dirs = []
            for dirpath, dirnames, filenames in os.walk(tmpdir):
                if "configure" in filenames:
                    config_dirs.append(dirpath)
            if not config_dirs:
                return None

            root_config = None
            for d in config_dirs:
                if os.path.isdir(os.path.join(d, "binutils")):
                    root_config = d
                    break
            if root_config is None:
                root_config = config_dirs[0]

            config_path = os.path.join(root_config, "configure")
            if not os.path.isfile(config_path):
                return None
            # Ensure configure is executable
            try:
                st = os.stat(config_path)
                os.chmod(config_path, st.st_mode | 0o111)
            except Exception:
                pass

            env = os.environ.copy()
            extra_cflags = "-g -O1 -fsanitize=address -fno-omit-frame-pointer"
            extra_ldflags = "-fsanitize=address"
            env["CFLAGS"] = (env.get("CFLAGS", "") + " " + extra_cflags).strip()
            env["CXXFLAGS"] = (env.get("CXXFLAGS", "") + " " + extra_cflags).strip()
            env["LDFLAGS"] = (env.get("LDFLAGS", "") + " " + extra_ldflags).strip()

            # Run configure
            try:
                proc = subprocess.run(
                    ["bash", "-c", "./configure --disable-werror"],
                    cwd=root_config,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
                if proc.returncode != 0:
                    return None
            except Exception:
                return None

            # Build objdump
            try:
                proc = subprocess.run(
                    ["make", "objdump", "-j4"],
                    cwd=root_config,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                )
                if proc.returncode != 0:
                    return None
            except Exception:
                return None

            # Locate objdump binary
            objdump_path = os.path.join(root_config, "binutils", "objdump")
            if not (os.path.isfile(objdump_path) and os.access(objdump_path, os.X_OK)):
                found = None
                for dirpath, dirnames, filenames in os.walk(root_config):
                    if "objdump" in filenames:
                        candidate = os.path.join(dirpath, "objdump")
                        if os.access(candidate, os.X_OK):
                            found = candidate
                            break
                if not found:
                    return None
                objdump_path = found

            input_path = os.path.join(root_config, "poc.bin")

            max_trials = 5000
            max_fuzz_time = 20.0
            start_time = time.time()

            for _ in range(max_trials):
                if time.time() - start_time > max_fuzz_time:
                    break

                # Try mostly 10-byte inputs (ground-truth length), but occasionally vary
                if _ % 10 == 0:
                    length = 16
                else:
                    length = 10

                data = os.urandom(length)
                try:
                    with open(input_path, "wb") as f:
                        f.write(data)
                except Exception:
                    continue

                try:
                    proc = subprocess.run(
                        [objdump_path, "-b", "binary", "-m", "tic30", "-D", input_path],
                        cwd=root_config,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=3,
                    )
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue

                stderr = proc.stderr or b""
                # Detect ASAN / UBSan / crash
                if (
                    b"ERROR: AddressSanitizer" in stderr
                    or b"AddressSanitizer" in stderr
                    or b"runtime error" in stderr
                    or b"Shadow bytes" in stderr
                    or proc.returncode < 0
                ):
                    return data

            return None

    def _fallback_poc(self) -> bytes:
        # Deterministic pseudo-random 64KB buffer to increase chance of triggering the bug
        length = 64 * 1024
        res = bytearray(length)
        x = 0x12345678
        for i in range(length):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            res[i] = x & 0xFF
        return bytes(res)

    def solve(self, src_path: str) -> bytes:
        try:
            poc = self._try_objdump_fuzz(src_path)
            if poc is not None:
                return poc
        except Exception:
            pass
        return self._fallback_poc()