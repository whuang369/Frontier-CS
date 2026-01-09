import os
import sys
import tarfile
import subprocess
import shutil
import time
import random
import glob
import tempfile
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            if src_path.endswith('.tar.gz') or src_path.endswith('.tgz'):
                with tarfile.open(src_path, "r:gz") as tar:
                    tar.extractall(path=work_dir)
            elif src_path.endswith('.tar.xz'):
                with tarfile.open(src_path, "r:xz") as tar:
                    tar.extractall(path=work_dir)
            elif src_path.endswith('.tar'):
                with tarfile.open(src_path, "r:") as tar:
                    tar.extractall(path=work_dir)
            else:
                # Try to work with whatever is there
                pass

            src_root = work_dir
            # Locate the actual source root (contains configure.ac or meson.build)
            for root, dirs, files in os.walk(work_dir):
                if "configure.ac" in files or "meson.build" in files:
                    src_root = root
                    break

            # 2. Build the vulnerable binary (ots-sanitize) with ASAN
            env = os.environ.copy()
            env["CC"] = "clang"
            env["CXX"] = "clang++"
            env["CFLAGS"] = "-fsanitize=address -g"
            env["CXXFLAGS"] = "-fsanitize=address -g"
            env["LDFLAGS"] = "-fsanitize=address"

            exe_path = None

            # Try Meson build
            if os.path.exists(os.path.join(src_root, "meson.build")):
                build_dir = os.path.join(src_root, "build_meson")
                try:
                    subprocess.run(["meson", "setup", build_dir], cwd=src_root, env=env, 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["ninja", "-C", build_dir], cwd=src_root, env=env, 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    found = glob.glob(os.path.join(build_dir, "**", "ots-sanitize"), recursive=True)
                    if found:
                        exe_path = found[0]
                except Exception:
                    pass

            # Try Autotools build
            if not exe_path and os.path.exists(os.path.join(src_root, "configure.ac")):
                try:
                    if os.path.exists(os.path.join(src_root, "autogen.sh")):
                        subprocess.run(["./autogen.sh"], cwd=src_root, env=env, 
                                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["./configure"], cwd=src_root, env=env, 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["make", "-j8"], cwd=src_root, env=env, 
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    found = glob.glob(os.path.join(src_root, "**", "ots-sanitize"), recursive=True)
                    # Prefer binary over libtool script if possible, but wrappers work
                    valid = [f for f in found if ".libs" not in f]
                    if valid:
                        exe_path = valid[0]
                    elif found:
                        exe_path = found[0]
                except Exception:
                    pass

            # Fallback search
            if not exe_path:
                found = glob.glob(os.path.join(work_dir, "**", "ots-sanitize"), recursive=True)
                if found:
                    exe_path = found[0]

            # 3. Collect Seeds (valid fonts from tests)
            seeds = []
            for ext in ["ttf", "otf", "woff", "woff2"]:
                for f in glob.glob(os.path.join(src_root, "**", f"*.{ext}"), recursive=True):
                    try:
                        with open(f, "rb") as fp:
                            data = fp.read()
                            if 0 < len(data) < 100 * 1024: # Limit size for speed
                                seeds.append(data)
                    except:
                        pass
                    if len(seeds) > 50: break
            
            if not seeds:
                # Minimal seed if none found
                seeds.append(b"\x00\x01\x00\x00\x00\x0c\x00\x80\x00\x03\x00\x10" + b"\x00" * 800)

            # 4. Fuzzing Loop
            if exe_path:
                start_time = time.time()
                # Run for up to 90 seconds (leaving buffer for setup/teardown)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = []
                    while time.time() - start_time < 90:
                        # Check completed
                        done = [f for f in futures if f.done()]
                        for f in done:
                            futures.remove(f)
                            res = f.result()
                            if res:
                                return res

                        # Fill pool
                        if len(futures) < 16:
                            futures.append(pool.submit(self._fuzz_task, exe_path, seeds, env, work_dir))
                        else:
                            time.sleep(0.05)
            
            # If no crash found or build failed, return a valid seed (does not crash fixed, might crash vulnerable)
            return seeds[0]

        except Exception:
            return b"\x00" * 800
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _fuzz_task(self, exe, seeds, env, wd):
        try:
            seed = random.choice(seeds)
            arr = bytearray(seed)
            if not arr: return None
            
            # Mutation strategy: mainly bitflips and byte overwrites to preserve structure
            num_mutations = random.randint(1, 6)
            for _ in range(num_mutations):
                idx = random.randint(0, len(arr) - 1)
                choice = random.randint(0, 2)
                if choice == 0: # Bit flip
                    arr[idx] ^= (1 << random.randint(0, 7))
                elif choice == 1: # Byte set
                    arr[idx] = random.randint(0, 255)
                elif choice == 2: # Arithmetic
                    val = arr[idx]
                    val = (val + random.choice([1, -1, 10, -10])) % 256
                    arr[idx] = val

            payload = bytes(arr)
            tmp_name = os.path.join(wd, f"fuzz_{random.randint(0, 100000000)}.ttf")
            
            with open(tmp_name, "wb") as f:
                f.write(payload)
            
            # Run target
            # ots-sanitize [file]
            p = subprocess.run([exe, tmp_name], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2, env=env)
            
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

            if p.returncode != 0:
                err = p.stderr.decode(errors='ignore')
                # Check for Heap Use After Free
                if "AddressSanitizer" in err and "heap-use-after-free" in err:
                    return payload
        except Exception:
            pass
        return None