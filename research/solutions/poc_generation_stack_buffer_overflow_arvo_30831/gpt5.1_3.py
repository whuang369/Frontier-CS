import os
import tarfile
import tempfile
import subprocess
import time
import random
import stat
from typing import List, Optional

ASAN_KEYWORDS = [
    "AddressSanitizer",
    "runtime error",
    "stack-buffer-overflow",
    "heap-buffer-overflow",
    "stack-buffer-underflow",
    "heap-use-after-free",
    "double free",
    "corrupted",
    "SIGSEGV",
    "Segmentation fault",
    "stack smashing detected",
]


def is_within_directory(directory: str, target: str) -> bool:
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            continue
    tar.extractall(path)


def extract_tarball(src_path: str, dst_dir: str) -> None:
    with tarfile.open(src_path, "r:*") as tar:
        safe_extract(tar, dst_dir)


def find_build_script(root_dir: str) -> Optional[str]:
    # Prefer a top-level build.sh if present
    top_level = os.path.join(root_dir, "build.sh")
    if os.path.isfile(top_level):
        return top_level
    for dirpath, _, filenames in os.walk(root_dir):
        if "build.sh" in filenames:
            return os.path.join(dirpath, "build.sh")
    return None


def build_project(root_dir: str, timeout: float = 50.0) -> None:
    script = find_build_script(root_dir)
    if script is not None:
        try:
            subprocess.run(
                ["bash", script],
                cwd=os.path.dirname(script),
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass

    # Fallback to make if a Makefile exists
    makefile = os.path.join(root_dir, "Makefile")
    if os.path.isfile(makefile):
        try:
            subprocess.run(
                ["make", "-j4"],
                cwd=root_dir,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass


def find_candidate_binaries(root_dir: str) -> List[str]:
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".git")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not os.access(path, os.X_OK):
                continue
            if any(
                name.endswith(ext)
                for ext in (
                    ".sh",
                    ".py",
                    ".pl",
                    ".rb",
                    ".so",
                    ".a",
                    ".o",
                    ".lo",
                    ".dll",
                    ".dylib",
                )
            ):
                continue
            try:
                with open(path, "rb") as f:
                    magic = f.read(4)
                if magic != b"\x7fELF":
                    continue
            except OSError:
                continue

            score = 0.0
            lower = name.lower()
            if "fuzz" in lower or "poc" in lower or "harness" in lower:
                score += 3.0
            if "coap" in lower:
                score += 2.0
            if "test" in lower or "demo" in lower or "example" in lower:
                score += 1.0
            rel_depth = len(os.path.relpath(path, root_dir).split(os.sep))
            score -= 0.01 * rel_depth
            candidates.append((score, path))
    candidates.sort(reverse=True)
    return [p for _, p in candidates]


def is_crash(proc: subprocess.CompletedProcess) -> bool:
    rc = proc.returncode
    out = proc.stdout + proc.stderr
    try:
        text = out.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if rc < 0:
        return True
    if rc != 0:
        for k in ASAN_KEYWORDS:
            if k in text:
                return True
    return False


def run_candidate(binary: str, data: bytes, timeout: float = 0.2) -> bool:
    # Try feeding via stdin
    try:
        proc = subprocess.run(
            [binary],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if is_crash(proc):
            return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    # Try via file argument
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tf:
            tf.write(data)
            tf.flush()
            proc = subprocess.run(
                [binary, tf.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            if is_crash(proc):
                return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    return False


def generate_initial_seeds() -> List[bytes]:
    seeds: List[bytes] = []
    seeds.append(b"")
    seeds.append(b"A" * 4)
    seeds.append(b"A" * 21)
    seeds.append(b"A" * 100)
    seeds.append(b"\x00" * 16)
    seeds.append(bytes(range(1, 22)))
    seeds.append(os.urandom(21))

    # CoAP-like seeds
    header1 = bytes([0x40, 0x01, 0x00, 0x01])  # CON, GET, ID=1
    header2 = bytes([0x44, 0x02, 0x00, 0x10])  # CON, POST, ID=16
    seeds.append(header1)
    seeds.append(header2)
    seeds.append(header1 + b"\xff" * 8)
    seeds.append(header1 + b"\xDF" + b"\xFF" * 16)
    seeds.append(header2 + b"\xFF" * 32)
    return seeds


def mutate(data: bytes, max_len: int = 64) -> bytes:
    if not data:
        return os.urandom(random.randint(1, max_len))
    if random.random() < 0.1:
        return os.urandom(random.randint(1, max_len))
    ba = bytearray(data)
    num_mut = random.randint(1, 8)
    for _ in range(num_mut):
        op = random.randint(0, 3)
        if op == 0 and len(ba) > 0:
            idx = random.randrange(len(ba))
            ba[idx] ^= 1 << random.randint(0, 7)
        elif op == 1 and len(ba) > 0:
            idx = random.randrange(len(ba))
            ba[idx] = random.randint(0, 255)
        elif op == 2 and len(ba) < max_len:
            idx = random.randrange(len(ba) + 1)
            ba.insert(idx, random.randint(0, 255))
        elif op == 3 and len(ba) > 1:
            idx = random.randrange(len(ba))
            del ba[idx]
    if len(ba) > max_len:
        ba = ba[:max_len]
    return bytes(ba)


def minimize_input(binary: str, data: bytes, time_budget: float) -> bytes:
    start = time.time()
    cand = data
    attempts = 0
    while (
        len(cand) > 1
        and time.time() - start < time_budget
        and attempts < 200
        and len(cand) > 21
    ):
        attempts += 1
        idx = random.randrange(len(cand))
        new = cand[:idx] + cand[idx + 1 :]
        if not new:
            continue
        if run_candidate(binary, new):
            cand = new
    return cand


def find_crashing_input_for_binary(binary: str, overall_time_budget: float) -> Optional[bytes]:
    start = time.time()
    seeds = generate_initial_seeds()
    corpus: List[bytes] = []

    for s in seeds:
        if time.time() - start > overall_time_budget:
            return None
        if run_candidate(binary, s):
            remaining = overall_time_budget - (time.time() - start)
            if remaining > 0 and len(s) > 21:
                s = minimize_input(binary, s, remaining)
            return s
        corpus.append(s)

    max_iters = 400
    i = 0
    while time.time() - start < overall_time_budget and i < max_iters:
        i += 1
        base = random.choice(corpus) if corpus else os.urandom(8)
        cand = mutate(base)
        if run_candidate(binary, cand):
            remaining = overall_time_budget - (time.time() - start)
            if remaining > 0 and len(cand) > 21:
                cand = minimize_input(binary, cand, remaining)
            return cand
        corpus.append(cand)
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        random.seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_tarball(src_path, tmpdir)
            build_project(tmpdir)

            binaries = find_candidate_binaries(tmpdir)
            total_budget = 25.0
            start = time.time()
            for binary in binaries:
                time_left = total_budget - (time.time() - start)
                if time_left <= 2.0:
                    break
                budget_per = min(10.0, max(1.0, time_left - 1.0))
                poc = find_crashing_input_for_binary(binary, budget_per)
                if poc is not None:
                    return poc

        # Fallback PoC guess if analysis/fuzzing fails
        return b"A" * 21