import os
import tarfile
import tempfile
import subprocess
import shutil
import glob
import re
import stat


def extract_tarball(src_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(src_path, "r:*") as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonpath([abs_directory, abs_target])
            return prefix == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not is_within_directory(dest_dir, member_path):
                continue
        tar.extractall(dest_dir)


def find_build_sh(base_dir: str) -> str | None:
    for root, _, files in os.walk(base_dir):
        if "build.sh" in files:
            return os.path.join(root, "build.sh")
    return None


def deduce_src_env(build_sh: str, base_dir: str) -> str:
    build_dir = os.path.dirname(build_sh)
    src_env = build_dir
    try:
        text = open(build_sh, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return src_env

    m = re.search(r"\$SRC/([A-Za-z0-9_\-./]+)", text)
    if not m:
        return src_env

    first_path = m.group(1)
    first_seg = first_path.split("/")[0]

    candidates = []
    for root, dirs, _ in os.walk(base_dir):
        if first_seg in dirs:
            candidates.append(root)

    best = None
    build_dir_abs = os.path.abspath(build_dir)
    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if build_dir_abs.startswith(cand_abs):
            if best is None or len(cand_abs) > len(best):
                best = cand_abs

    if best:
        return best
    return src_env


def build_fuzzers(build_sh: str, tmpdir: str) -> str | None:
    env = os.environ.copy()
    src_env = deduce_src_env(build_sh, tmpdir)
    env["SRC"] = src_env

    out_dir = os.path.join(tmpdir, "out")
    os.makedirs(out_dir, exist_ok=True)
    env["OUT"] = out_dir

    env.setdefault("CC", "clang")
    env.setdefault("CXX", "clang++")
    env.setdefault("FUZZING_ENGINE", "libfuzzer")
    env.setdefault("SANITIZER", "address")
    env.setdefault("ARCHITECTURE", "x86_64")
    env.setdefault("LIB_FUZZING_ENGINE", "-fsanitize=fuzzer,address")

    # Keep CFLAGS/CXXFLAGS modest; build.sh may append sanitizers/options.
    env.setdefault("CFLAGS", "-g -O1 -fno-omit-frame-pointer")
    env.setdefault("CXXFLAGS", "-g -O1 -fno-omit-frame-pointer")

    log_path = os.path.join(tmpdir, "build.log")
    try:
        with open(log_path, "wb") as log:
            subprocess.run(
                ["bash", build_sh],
                cwd=os.path.dirname(build_sh),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
    except Exception:
        return None
    return out_dir


def find_fuzzers(out_dir: str) -> list[str]:
    fuzzers: list[str] = []
    for root, _, files in os.walk(out_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not (st.st_mode & stat.S_IXUSR):
                continue
            lower = name.lower()
            if lower.endswith(".a") or lower.endswith(".so") or lower.endswith(".o"):
                continue
            fuzzers.append(path)
    return fuzzers


def choose_fuzzer(fuzzers: list[str]) -> str:
    if not fuzzers:
        raise ValueError("No fuzzers found")

    def score(path: str) -> int:
        name = os.path.basename(path).lower()
        s = 0
        if "rar" in name:
            s -= 4
        if "archive" in name:
            s -= 2
        if "read" in name:
            s -= 1
        return s

    fuzzers_sorted = sorted(fuzzers, key=score)
    return fuzzers_sorted[0]


def collect_rar_seeds(base_dir: str, dest_dir: str) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    rar5_sig = b"Rar!\x1a\x07\x01\x00"
    rar4_sig = b"Rar!\x1a\x07\x00"
    rar_files: list[tuple[str, int]] = []

    for root, _, files in os.walk(base_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith(".rar") or lower.endswith(".rar5") or "rar5" in lower:
                src_path = os.path.join(root, name)
                try:
                    size = os.path.getsize(src_path)
                except OSError:
                    continue
                if size == 0 or size > 2 * 1024 * 1024:
                    continue
                rar_files.append((src_path, size))

    # Prefer smaller files
    rar_files.sort(key=lambda x: x[1])

    # First collect true RAR5, then fall back to any RAR if none
    rar5_candidates: list[str] = []
    other_candidates: list[str] = []

    for src_path, _ in rar_files:
        try:
            with open(src_path, "rb") as f:
                sig = f.read(8)
        except OSError:
            continue
        if sig == rar5_sig:
            rar5_candidates.append(src_path)
        elif sig.startswith(rar4_sig[:7]) or sig.startswith(b"Rar!"):
            other_candidates.append(src_path)

    chosen_list = rar5_candidates if rar5_candidates else other_candidates

    for src_path in chosen_list:
        dst = os.path.join(dest_dir, f"seed_{count}.rar")
        try:
            shutil.copy2(src_path, dst)
            count += 1
        except OSError:
            continue

    return count


def run_fuzzer_and_get_crash(
    fuzzer: str, seeds_dir: str | None, work_dir: str, max_total_time: int = 60
) -> bytes | None:
    artifacts_dir = os.path.join(work_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    cmd = [
        fuzzer,
        f"-max_total_time={max_total_time}",
        "-timeout=10",
        f"-artifact_prefix={artifacts_dir}/",
        "-jobs=1",
        "-workers=1",
    ]

    if seeds_dir is not None:
        try:
            if any(os.scandir(seeds_dir)):
                cmd.append(seeds_dir)
        except FileNotFoundError:
            pass

    env = os.environ.copy()
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0")

    try:
        subprocess.run(
            cmd,
            cwd=os.path.dirname(fuzzer),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=max_total_time + 20,
        )
    except Exception:
        return None

    crashes = sorted(
        glob.glob(os.path.join(artifacts_dir, "crash-*")), key=lambda p: os.path.getsize(p)
    )
    if not crashes:
        return None

    crash_path = crashes[0]
    try:
        with open(crash_path, "rb") as f:
            return f.read()
    except OSError:
        return None


def fallback_poc() -> bytes:
    # Simple RAR5-like header with large body; may or may not trigger, but used only as fallback.
    header = b"Rar!\x1a\x07\x01\x00"
    body = b"\x00" * 2048
    return header + body


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="poc_gen_")
        try:
            extract_tarball(src_path, tmpdir)
            build_sh = find_build_sh(tmpdir)
            if build_sh is not None:
                out_dir = build_fuzzers(build_sh, tmpdir)
                if out_dir is not None:
                    fuzzers = find_fuzzers(out_dir)
                    if fuzzers:
                        fuzzer = choose_fuzzer(fuzzers)
                        seeds_dir = os.path.join(tmpdir, "seeds")
                        collect_rar_seeds(tmpdir, seeds_dir)
                        poc = run_fuzzer_and_get_crash(
                            fuzzer, seeds_dir, tmpdir, max_total_time=60
                        )
                        if poc:
                            return poc
        except Exception:
            pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return fallback_poc()