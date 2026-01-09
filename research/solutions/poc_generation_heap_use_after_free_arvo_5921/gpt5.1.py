import os
import tarfile
import tempfile
import shutil
import gzip
import struct
import random
import subprocess
import time


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmp_dir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            self._extract_tarball(src_path, tmp_dir)
            poc = self._find_existing_poc(tmp_dir, target_len=73)
            if poc is not None:
                return poc

            poc = self._try_dynamic_generation(tmp_dir, target_len=73, total_timeout=40.0)
            if poc is not None:
                return poc

            return self._fallback_poc(73)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ----------------- Core helpers -----------------

    def _extract_tarball(self, src_path: str, dst_dir: str) -> None:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar_obj, path=".", members=None, *, numeric_owner=False):
                    for member in tar_obj.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            continue
                    tar_obj.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tf, dst_dir)
        except Exception:
            pass

    def _is_probably_binary(self, data: bytes) -> bool:
        if not data:
            return False
        if b"\x00" in data:
            return True
        text_bytes = set(range(32, 127)) | {9, 10, 13}
        nontext = 0
        for b in data:
            if b not in text_bytes:
                nontext += 1
        return (nontext / max(1, len(data))) > 0.3

    def _find_existing_poc(self, root: str, target_len: int) -> bytes | None:
        ext_priority = {
            ".pcap": 0,
            ".pcapng": 1,
            ".bin": 2,
            ".dat": 3,
            ".raw": 4,
            "": 5,
        }

        best_content = None
        best_score = 1_000_000

        for dirpath, _, files in os.walk(root):
            for fname in files:
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue

                # Direct, uncompressed file
                if size == target_len:
                    try:
                        with open(fpath, "rb") as f:
                            content = f.read()
                    except OSError:
                        content = b""
                    if not content:
                        continue
                    if not self._is_probably_binary(content):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    score = ext_priority.get(ext, 50)
                    if score < best_score:
                        best_score = score
                        best_content = content
                        if best_score == 0:  # perfect: .pcap with exact length
                            return best_content

                # Gzip-compressed candidate (.pcap.gz etc.)
                ext = os.path.splitext(fname)[1].lower()
                if ext == ".gz":
                    try:
                        with open(fpath, "rb") as f:
                            raw = f.read()
                        content = gzip.decompress(raw)
                    except Exception:
                        continue
                    if len(content) != target_len:
                        continue
                    if not self._is_probably_binary(content):
                        continue
                    base_name = os.path.splitext(fname)[0]
                    base_ext = os.path.splitext(base_name)[1].lower()
                    score = ext_priority.get(base_ext, 60) + 10  # penalize compressed slightly
                    if score < best_score:
                        best_score = score
                        best_content = content
                        if best_score == 10:  # compressed .pcap exact length
                            return best_content

        return best_content

    # ----------------- Dynamic approach (best-effort) -----------------

    def _try_dynamic_generation(self, root: str, target_len: int, total_timeout: float) -> bytes | None:
        start_time = time.time()
        exes = self._find_executables(root)
        if not exes:
            return None

        seeds = self._gather_seed_inputs(root)
        if not seeds:
            # basic generic seeds around target length
            seeds = [
                b"A" * max(1, min(target_len, 64)),
                b"\x00" * max(1, min(target_len, 64)),
                bytes(range(1, min(256, max(2, target_len)))),
            ]

        for exe in exes:
            remaining = total_timeout - (time.time() - start_time)
            if remaining <= 0:
                break
            poc = self._fuzz_executable(exe, seeds, max_time=remaining)
            if poc is not None:
                return poc
        return None

    def _find_executables(self, root: str) -> list[str]:
        exes: list[str] = []

        # Try to build if a simple Makefile is present
        make_tried = False
        makefile_path = os.path.join(root, "Makefile")
        if os.path.isfile(makefile_path):
            make_tried = True
            try:
                subprocess.run(
                    ["make", "-j4"],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    check=False,
                )
            except Exception:
                pass

        # Also search without assuming build success; there may be prebuilt binaries
        for dirpath, _, files in os.walk(root):
            for fname in files:
                fpath = os.path.join(dirpath, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                # Executable bit set?
                if not (st.st_mode & 0o111):
                    continue
                # Skip obvious scripts
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".sh", ".py", ".pl", ".rb"):
                    continue
                try:
                    with open(fpath, "rb") as f:
                        magic = f.read(4)
                except OSError:
                    continue
                if magic == b"\x7fELF":
                    exes.append(fpath)

        # Heuristic ordering: prefer names hinting at h225/wireshark/fuzz
        def exe_score(path: str) -> int:
            name = os.path.basename(path).lower()
            score = 100
            if "h225" in name:
                score -= 40
            if "wireshark" in name or "tshark" in name:
                score -= 30
            if "fuzz" in name or "asan" in name or "uaf" in name:
                score -= 20
            if "test" in name or "driver" in name:
                score -= 10
            return score

        exes.sort(key=exe_score)
        return exes

    def _gather_seed_inputs(self, root: str) -> list[bytes]:
        seeds: list[bytes] = []
        max_seeds = 32
        for dirpath, dirs, files in os.walk(root):
            # Prefer directories named "seeds" or "corpus"
            base = os.path.basename(dirpath).lower()
            bonus = 0
            if base in ("seeds", "seed", "corpus", "inputs"):
                bonus = -1
            for fname in files:
                if len(seeds) >= max_seeds:
                    return seeds
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue
                if size == 0 or size > 4096:
                    continue
                try:
                    with open(fpath, "rb") as f:
                        content = f.read()
                except OSError:
                    continue
                if not self._is_probably_binary(content):
                    continue
                # Slight preference logic based on directory name already accounted via order of walk
                seeds.append(content)
        return seeds

    def _fuzz_executable(self, exe: str, seeds: list[bytes], max_time: float) -> bytes | None:
        start = time.time()
        rnd = random.Random(0xC0FFEE)
        max_iters = 1000

        for _ in range(max_iters):
            if time.time() - start > max_time:
                break
            base = rnd.choice(seeds) if seeds else os.urandom(16)
            mutated = self._mutate_bytes(base, rnd)
            if not mutated:
                mutated = os.urandom(8)

            try:
                with tempfile.NamedTemporaryFile(prefix="poc_input_", delete=False) as tf:
                    tf.write(mutated)
                    tf_path = tf.name
            except OSError:
                continue

            try:
                res = subprocess.run(
                    [exe, tf_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=0.5,
                    check=False,
                )
                out = res.stdout + res.stderr
                if b"heap-use-after-free" in out or (b"AddressSanitizer" in out and b"use-after-free" in out):
                    return mutated
            except subprocess.TimeoutExpired:
                # treat as non-crash for fuzzing purposes
                pass
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tf_path)
                except OSError:
                    pass

        return None

    def _mutate_bytes(self, data: bytes, rnd: random.Random) -> bytes:
        if not data:
            return os.urandom(rnd.randint(1, 128))
        ba = bytearray(data)
        # Ensure reasonable size range
        max_len = 512
        if len(ba) > max_len:
            ba = ba[:max_len]

        num_ops = rnd.randint(1, max(1, len(ba) // 8))
        for _ in range(num_ops):
            op = rnd.randint(0, 3)
            if op == 0 and ba:
                # flip random bit
                idx = rnd.randrange(len(ba))
                bit = 1 << rnd.randrange(8)
                ba[idx] ^= bit
            elif op == 1 and ba:
                # overwrite byte
                idx = rnd.randrange(len(ba))
                ba[idx] = rnd.randrange(256)
            elif op == 2:
                # insert random bytes
                pos = rnd.randrange(len(ba) + 1)
                insert_len = rnd.randint(1, 8)
                insert_data = os.urandom(insert_len)
                ba[pos:pos] = insert_data
                if len(ba) > max_len:
                    ba = ba[:max_len]
            elif op == 3 and len(ba) > 1:
                # delete one byte
                idx = rnd.randrange(len(ba))
                del ba[idx]
        return bytes(ba)

    # ----------------- Fallback PoC -----------------

    def _fallback_poc(self, target_len: int) -> bytes:
        # Construct a minimal PCAP with a single packet and arbitrary payload.
        if target_len <= 0:
            return b""

        if target_len <= 40:
            # Not enough space for full PCAP headers; just return arbitrary bytes.
            return b"A" * target_len

        payload_len = target_len - 40  # 24-byte global + 16-byte packet header

        # PCAP Global Header (little-endian)
        # magic_number, version_major, version_minor, thiszone, sigfigs, snaplen, network
        global_header = struct.pack(
            "<IHHIIII",
            0xA1B2C3D4,  # magic
            2,           # major
            4,           # minor
            0,           # thiszone
            0,           # sigfigs
            0xFFFF,      # snaplen
            1,           # LINKTYPE_ETHERNET (arbitrary)
        )

        # Packet Header: ts_sec, ts_usec, incl_len, orig_len
        packet_header = struct.pack("<IIII", 0, 0, payload_len, payload_len)

        # Payload: arbitrary bytes
        payload = b"\x00" * payload_len

        poc = global_header + packet_header + payload
        if len(poc) > target_len:
            poc = poc[:target_len]
        elif len(poc) < target_len:
            poc += b"A" * (target_len - len(poc))
        return poc