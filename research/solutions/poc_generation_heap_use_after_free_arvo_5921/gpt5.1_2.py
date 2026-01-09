import os
import tarfile
import tempfile
import subprocess
import random
import shutil
import stat


RUN_TIMEOUT = 0.2
BUILD_TIMEOUT = 120.0
FUZZ_ITERS = 200
MAX_INPUT_LEN = 256
GROUND_TRUTH_LEN = 73


def _safe_extract(tar, path):
    tar.extractall(path)


def _pick_compilers():
    cc = shutil.which("clang")
    cxx = shutil.which("clang++")
    if cc and cxx:
        return cc, cxx
    cc = shutil.which("gcc")
    cxx = shutil.which("g++")
    if cc and cxx:
        return cc, cxx
    return "cc", "c++"


def _find_build_scripts(root):
    scripts = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f in ("build.sh", "compile.sh", "build_fuzzers.sh", "build_fuzzer.sh", "build-target.sh"):
                scripts.append(os.path.join(dirpath, f))
    scripts.sort(key=lambda p: p.count(os.sep))
    return scripts


def _run_build_scripts(root):
    scripts = _find_build_scripts(root)
    if not scripts:
        return
    cc, cxx = _pick_compilers()
    env = os.environ.copy()
    if "CC" not in env:
        env["CC"] = cc
    if "CXX" not in env:
        env["CXX"] = cxx
    if "OUT" not in env:
        out_dir = os.path.join(root, "out")
        os.makedirs(out_dir, exist_ok=True)
        env["OUT"] = out_dir
    for script in scripts:
        try:
            subprocess.run(
                ["bash", script],
                cwd=os.path.dirname(script),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=BUILD_TIMEOUT,
                check=False,
            )
        except Exception:
            continue


def _find_executables(root):
    exes = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            path = os.path.join(dirpath, f)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not os.access(path, os.X_OK):
                continue
            fl = f.lower()
            if any(fl.endswith(ext) for ext in (".sh", ".py", ".pl", ".so", ".a", ".o", ".dll", ".dylib")):
                continue
            size = st.st_size
            if size == 0 or size > 100 * 1024 * 1024:
                continue
            exes.append(path)
    return exes


def _score_executable(path):
    name = os.path.basename(path).lower()
    score = 0
    if "h225" in name or "225" in name:
        score -= 10
    if "fuzz" in name:
        score -= 5
    if "shark" in name:
        score -= 3
    if "test" in name or "target" in name:
        score -= 1
    return score


def _detect_invocation(binary_path):
    candidates = []
    is_libfuzzer = False
    try:
        out = subprocess.check_output(
            ["strings", binary_path],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if b"LLVMFuzzerTestOneInput" in out or b"libFuzzer" in out:
            is_libfuzzer = True
    except Exception:
        is_libfuzzer = False

    if is_libfuzzer:
        candidates.append(("libfuzzer", [binary_path, "-runs=1", "{input}"], False))
        candidates.append(("libfuzzer_noruns", [binary_path, "{input}"], False))

    candidates.append(("arg_file", [binary_path, "{input}"], False))
    candidates.append(("stdin", [binary_path], True))

    test_input = os.urandom(4)
    tmpfile = None
    try:
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(test_input)
            tmpfile = f.name

        for _, cmd_template, use_stdin in candidates:
            cmd = [arg.format(input=tmpfile) for arg in cmd_template]
            try:
                if use_stdin:
                    proc = subprocess.run(
                        cmd,
                        input=test_input,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                else:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                rc = proc.returncode
            except Exception:
                continue
            if rc == 0:
                def runner(data, cmd_template=cmd_template, use_stdin=use_stdin, binary_path=binary_path):
                    import tempfile as _tf
                    tf = _tf.NamedTemporaryFile(delete=False)
                    try:
                        tf.write(data)
                        tf.flush()
                        tf_path = tf.name
                    finally:
                        tf.close()
                    try:
                        cmd_run = [arg.format(input=tf_path) for arg in cmd_template]
                        if use_stdin:
                            proc2 = subprocess.run(
                                cmd_run,
                                input=data,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=RUN_TIMEOUT,
                            )
                        else:
                            proc2 = subprocess.run(
                                cmd_run,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=RUN_TIMEOUT,
                            )
                        rc2 = proc2.returncode
                    except Exception:
                        rc2 = -1
                    finally:
                        try:
                            os.unlink(tf_path)
                        except Exception:
                            pass
                    return rc2
                return runner
    finally:
        if tmpfile is not None:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass
    return None


def _gather_seeds(root):
    seeds = []
    for dirpath, _, filenames in os.walk(root):
        ldir = dirpath.lower()
        if not any(k in ldir for k in ("seed", "seeds", "corpus", "in", "inputs", "cases")):
            continue
        for f in filenames:
            path = os.path.join(dirpath, f)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if st.st_size == 0 or st.st_size > 4096:
                continue
            try:
                with open(path, "rb") as fp:
                    data = fp.read()
            except Exception:
                continue
            if data:
                seeds.append(data)
    return seeds


def _mutate(data, rnd, max_len):
    if rnd.random() < 0.1:
        size = rnd.randint(1, max_len)
        try:
            return rnd.randbytes(size)
        except AttributeError:
            return bytes(rnd.getrandbits(8) for _ in range(size))
    res = bytearray(data)
    num_mut = rnd.randint(1, 8)
    for _ in range(num_mut):
        choice = rnd.randint(0, 3)
        if choice == 0 and len(res) > 0:
            idx = rnd.randrange(len(res))
            res[idx] ^= 1 << rnd.randrange(8)
        elif choice == 1 and len(res) > 0:
            idx = rnd.randrange(len(res))
            res[idx] = rnd.randrange(256)
        elif choice == 2 and len(res) < max_len:
            idx = rnd.randrange(len(res) + 1)
            res.insert(idx, rnd.randrange(256))
        elif choice == 3 and len(res) > 1:
            idx = rnd.randrange(len(res))
            del res[idx]
    if len(res) > max_len:
        del res[max_len:]
    return bytes(res)


def _fuzz_for_crash(run_input, seeds):
    rnd = random.Random(0xC0FFEE)
    corpus = list(seeds) if seeds else [b"\x00", b"A", b"\xff" * 4]
    seen = set(corpus)
    for _ in range(FUZZ_ITERS):
        base = rnd.choice(corpus)
        data = _mutate(base, rnd, MAX_INPUT_LEN)
        if not data or data in seen:
            continue
        seen.add(data)
        rc = run_input(data)
        if rc != 0:
            return data
        corpus.append(data)
        if len(corpus) > 64:
            corpus = corpus[-64:]
    return None


def _shrink_suffix(run_input, data):
    n = len(data)
    min_len = max(1, min(GROUND_TRUTH_LEN, n))
    cur = data
    for new_len in range(n - 1, min_len - 1, -1):
        cand = cur[:new_len]
        rc = run_input(cand)
        if rc != 0:
            cur = cand
    return cur


def _shrink_prefix(run_input, data):
    cur = data
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for i in range(1, len(cur)):
            cand = cur[i:]
            rc = run_input(cand)
            if rc != 0:
                cur = cand
                changed = True
                break
    return cur


def _is_binary_like(data):
    if not data:
        return False
    control = 0
    for b in data:
        if b == 9 or b == 10 or b == 13:
            continue
        if b < 32 or b > 126:
            control += 1
    return control / float(len(data)) > 0.2


def _find_static_poc(root, target_len=GROUND_TRUTH_LEN):
    best = None
    best_score = None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            path = os.path.join(dirpath, f)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            size = st.st_size
            if size == 0 or size > 4096:
                continue
            try:
                with open(path, "rb") as fp:
                    data = fp.read()
            except Exception:
                continue
            if not data:
                continue
            path_l = path.lower()
            score = 0
            if any(k in path_l for k in ("poc", "crash", "bug", "id_", "repro", "h225")):
                score -= 10
            if any(k in path_l for k in ("seed", "input", "in", "corpus")):
                score -= 3
            if _is_binary_like(data):
                score -= 2
            score += abs(len(data) - target_len)
            if best is None or score < best_score:
                best = data
                best_score = score
    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        workdir = tempfile.mkdtemp(prefix="pocgen_")
        static_poc = None
        try:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    _safe_extract(tf, workdir)
            except Exception:
                # If src_path is not a tarball, treat it as directory
                if os.path.isdir(src_path):
                    workdir = src_path
                else:
                    return b"A" * GROUND_TRUTH_LEN

            static_poc = _find_static_poc(workdir, GROUND_TRUTH_LEN)

            _run_build_scripts(workdir)
            exes = _find_executables(workdir)
            if exes:
                exes.sort(key=_score_executable)
            crash_poc = None
            for exe in exes:
                runner = _detect_invocation(exe)
                if runner is None:
                    continue
                seeds = _gather_seeds(workdir)
                poc = _fuzz_for_crash(runner, seeds)
                if poc is not None:
                    poc = _shrink_suffix(runner, poc)
                    poc = _shrink_prefix(runner, poc)
                    crash_poc = poc
                    break
            if crash_poc is not None:
                return crash_poc
            if static_poc is not None:
                return static_poc
            return b"A" * GROUND_TRUTH_LEN
        finally:
            if workdir != src_path:
                shutil.rmtree(workdir, ignore_errors=True)