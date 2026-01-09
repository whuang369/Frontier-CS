import os
import re
import io
import sys
import time
import shlex
import tarfile
import random
import hashlib
import tempfile
import subprocess
from typing import List, Tuple, Optional, Dict, Set


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
            root = self._prepare_source(src_path, td)
            if not root or not os.path.isdir(root):
                return b"add a\nadd a\n"

            sources = self._collect_sources(root)
            if not sources:
                return b"add a\nadd a\n"

            src_text_concat = self._concat_small_sources(sources, max_total=4_000_000)
            seed = int.from_bytes(hashlib.sha256(src_text_concat).digest()[:8], "little", signed=False)
            rng = random.Random(seed)

            tokens = self._extract_string_tokens(src_text_concat)
            is_texty = self._looks_text_based(src_text_concat)

            compiler = self._find_compiler()
            if compiler is None:
                return self._fallback_payload(tokens)

            build_dir = os.path.join(td, "build")
            os.makedirs(build_dir, exist_ok=True)

            fuzz_entry_files = self._find_fuzzer_entry_files(sources)
            main_files = self._find_main_files(sources)

            binary_path = os.path.join(build_dir, "runner")
            run_mode = None

            if fuzz_entry_files:
                runner_cpp = os.path.join(build_dir, "runner_main.cpp")
                self._write_fuzzer_runner_main(runner_cpp)
                compile_sources = self._filter_out_main_files(sources)
                compile_sources.append(runner_cpp)
                ok = self._compile(compiler, binary_path, compile_sources, root)
                run_mode = "file_or_stdin"
                if not ok:
                    ok = self._compile(compiler, binary_path, compile_sources, root, extra_defines=["-D_GLIBCXX_ASSERTIONS=0"])
                if not ok:
                    return self._fallback_payload(tokens)
            else:
                if not main_files:
                    return self._fallback_payload(tokens)
                chosen_main = self._choose_main_file(main_files, src_text_concat)
                compile_sources = self._compile_sources_with_single_main(sources, chosen_main)
                ok = self._compile(compiler, binary_path, compile_sources, root)
                if not ok:
                    return self._fallback_payload(tokens)
                run_mode = "auto"

            def run_input(data: bytes) -> Tuple[int, bytes]:
                return self._run_binary(binary_path, data, run_mode, chosen_main if not fuzz_entry_files else None)

            start = time.monotonic()
            budget = 28.0

            candidates: List[bytes] = []
            candidates.extend(self._initial_candidates(tokens, is_texty))
            candidates.extend(self._format_guess_candidates(tokens))
            candidates.extend(self._basic_binary_candidates(tokens))

            seen: Set[bytes] = set()
            deduped: List[bytes] = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            candidates = deduped

            crashing = None
            crash_stderr = b""

            for c in candidates:
                if time.monotonic() - start > budget * 0.55:
                    break
                rc, err = run_input(c)
                if self._is_target_crash(rc, err):
                    crashing = c
                    crash_stderr = err
                    break

            if crashing is None:
                corpus = candidates[:]
                if not corpus:
                    corpus = [b"", b"\x00", b"A"]
                best = corpus[0]
                best_score = -1

                iters = 0
                while time.monotonic() - start < budget * 0.78:
                    iters += 1
                    base = rng.choice(corpus)
                    mutated = self._mutate(base, rng, tokens, prefer_text=is_texty, max_len=200)
                    if not mutated:
                        continue

                    rc, err = run_input(mutated)
                    if self._is_target_crash(rc, err):
                        crashing = mutated
                        crash_stderr = err
                        break

                    score = 0
                    if rc == 0:
                        score += 1
                    if b"exception" in err.lower():
                        score += 1
                    if b"invalid" in err.lower() or b"usage" in err.lower():
                        score -= 1
                    if score > best_score:
                        best_score = score
                        best = mutated
                        if len(corpus) < 64:
                            corpus.append(mutated)
                    elif len(corpus) < 16 and rng.random() < 0.05:
                        corpus.append(mutated)

            if crashing is None:
                return self._fallback_payload(tokens)

            time_left = budget - (time.monotonic() - start)
            if time_left > 2.5:
                crashing = self._minimize(crashing, run_input, start, budget, require_node_add=True)

            crashing = self._trim_trailing(crashing, run_input, start, budget)

            if len(crashing) == 0:
                crashing = b"add a\nadd a\n"

            return crashing

    def _prepare_source(self, src_path: str, td: str) -> Optional[str]:
        if os.path.isdir(src_path):
            return os.path.abspath(src_path)
        if not os.path.isfile(src_path):
            return None
        extract_dir = os.path.join(td, "src")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

                for m in tf.getmembers():
                    name = m.name
                    if name.startswith("/") or ".." in name.split("/"):
                        continue
                    target_path = os.path.join(extract_dir, name)
                    if not is_within_directory(extract_dir, target_path):
                        continue
                    tf.extract(m, path=extract_dir)
        except Exception:
            return None

        entries = [os.path.join(extract_dir, p) for p in os.listdir(extract_dir)]
        dirs = [p for p in entries if os.path.isdir(p)]
        if len(dirs) == 1:
            return dirs[0]
        return extract_dir

    def _collect_sources(self, root: str) -> List[str]:
        ex_dirs = {"build", "cmake-build-debug", "cmake-build-release", ".git", ".svn", ".hg", "out", "dist", "node_modules"}
        sources = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ex_dirs and not d.startswith(".")]
            low_dp = dirpath.lower()
            if any(x in low_dp for x in ("/test", "\\test", "/tests", "\\tests", "/benchmark", "\\benchmark")):
                continue
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith((".c", ".cc", ".cpp", ".cxx")):
                    sources.append(os.path.join(dirpath, fn))
        return sources

    def _concat_small_sources(self, sources: List[str], max_total: int = 2_000_000) -> bytes:
        buf = bytearray()
        total = 0
        for p in sources[:400]:
            try:
                st = os.stat(p)
                if st.st_size > 500_000:
                    continue
                with open(p, "rb") as f:
                    data = f.read(min(st.st_size, 600_000))
                if total + len(data) > max_total:
                    break
                buf.extend(data)
                buf.extend(b"\n")
                total += len(data) + 1
            except Exception:
                continue
        return bytes(buf)

    def _extract_string_tokens(self, data: bytes) -> List[bytes]:
        try:
            txt = data.decode("utf-8", errors="ignore")
        except Exception:
            return []
        raw = re.findall(r'"((?:\\.|[^"\\])*)"', txt)
        tokens: Set[bytes] = set()
        for s in raw:
            if not s:
                continue
            if len(s) > 64:
                continue
            try:
                u = bytes(s, "utf-8").decode("unicode_escape", errors="ignore")
            except Exception:
                u = s
            u = u.replace("\r", "")
            if not u:
                continue
            if any(ord(ch) < 9 for ch in u):
                continue
            try:
                b = u.encode("utf-8")
            except Exception:
                continue
            if 0 < len(b) <= 64:
                tokens.add(b)
        lst = sorted(tokens, key=lambda x: (-len(x), x))
        filtered: List[bytes] = []
        for t in lst:
            if len(filtered) >= 200:
                break
            if all(32 <= c < 127 for c in t) and not any(ch in t for ch in (b"\x7f",)):
                filtered.append(t)
        return filtered

    def _looks_text_based(self, data: bytes) -> bool:
        try:
            txt = data.decode("utf-8", errors="ignore").lower()
        except Exception:
            return False
        if "std::getline" in txt or "getline(" in txt:
            return True
        if "std::cin" in txt or "cin >>" in txt:
            return True
        if "scanf(" in txt or "fgets(" in txt:
            return True
        if "argv[1]" in txt and ("ifstream" in txt or "fopen" in txt):
            return True
        return False

    def _find_compiler(self) -> Optional[str]:
        for c in ("clang++", "g++"):
            p = shutil_which(c)
            if p:
                return p
        return None

    def _find_fuzzer_entry_files(self, sources: List[str]) -> List[str]:
        hits = []
        pat = re.compile(rb"\bLLVMFuzzerTestOneInput\b")
        for p in sources:
            try:
                with open(p, "rb") as f:
                    d = f.read(400_000)
                if pat.search(d):
                    hits.append(p)
            except Exception:
                continue
        return hits

    def _find_main_files(self, sources: List[str]) -> List[str]:
        hits = []
        pat = re.compile(rb"\bint\s+main\s*\(")
        for p in sources:
            try:
                with open(p, "rb") as f:
                    d = f.read(500_000)
                if pat.search(d):
                    hits.append(p)
            except Exception:
                continue
        return hits

    def _choose_main_file(self, main_files: List[str], src_text_concat: bytes) -> str:
        def score(p: str) -> int:
            lp = p.lower()
            s = 0
            if "/src/" in lp or "\\src\\" in lp:
                s += 3
            if lp.endswith(("main.cc", "main.cpp", "main.cxx", "main.c")):
                s += 3
            if "test" in lp or "bench" in lp:
                s -= 5
            try:
                with open(p, "rb") as f:
                    d = f.read(200_000)
                if b"usage" in d.lower():
                    s += 1
                if b"fuzz" in d.lower():
                    s -= 2
            except Exception:
                pass
            return s
        return sorted(main_files, key=lambda p: (-score(p), p))[0]

    def _filter_out_main_files(self, sources: List[str]) -> List[str]:
        mains = set(self._find_main_files(sources))
        out = []
        for p in sources:
            if p in mains:
                continue
            out.append(p)
        return out

    def _compile_sources_with_single_main(self, sources: List[str], chosen_main: str) -> List[str]:
        mains = set(self._find_main_files(sources))
        out = []
        for p in sources:
            if p in mains and p != chosen_main:
                continue
            out.append(p)
        return out

    def _write_fuzzer_runner_main(self, path: str) -> None:
        code = r'''
#include <cstdint>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

static std::vector<uint8_t> read_all(std::istream& in) {
    std::vector<uint8_t> buf;
    in.seekg(0, std::ios::end);
    std::streampos end = in.tellg();
    if (end > 0 && end < (std::streampos)(256 * 1024 * 1024)) {
        buf.resize((size_t)end);
        in.seekg(0, std::ios::beg);
        in.read((char*)buf.data(), (std::streamsize)buf.size());
        return buf;
    }
    in.clear();
    in.seekg(0, std::ios::beg);
    buf.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return buf;
}

int main(int argc, char** argv) {
    std::vector<uint8_t> data;
    if (argc > 1) {
        std::ifstream f(argv[1], std::ios::binary);
        if (!f) return 0;
        data = read_all(f);
    } else {
        std::cin.sync_with_stdio(false);
        std::cin.tie(nullptr);
        data = read_all(std::cin);
    }
    const uint8_t* ptr = data.empty() ? (const uint8_t*)"" : data.data();
    LLVMFuzzerTestOneInput(ptr, data.size());
    return 0;
}
'''
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

    def _compile(self, compiler: str, out_bin: str, sources: List[str], root: str, extra_defines: Optional[List[str]] = None) -> bool:
        include_dirs = self._collect_include_dirs(root)
        cmd = [compiler, "-std=c++17", "-O1", "-g", "-fno-omit-frame-pointer", "-fexceptions", "-w"]
        cmd += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
        cmd += ["-pthread"]
        if extra_defines:
            cmd += extra_defines
        for inc in include_dirs:
            cmd.append("-I" + inc)
        cmd += ["-o", out_bin]
        cmd += sources

        try:
            r = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if r.returncode == 0 and os.path.isfile(out_bin):
                return True
            return False
        except Exception:
            return False

    def _collect_include_dirs(self, root: str) -> List[str]:
        dirs = {root}
        for cand in ("include", "src", "inc"):
            p = os.path.join(root, cand)
            if os.path.isdir(p):
                dirs.add(p)
        # include dirs at depth <= 2 named include
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 2:
                dirnames[:] = []
                continue
            base = os.path.basename(dirpath).lower()
            if base in ("include", "inc"):
                dirs.add(dirpath)
        return sorted(dirs)

    def _run_binary(self, bin_path: str, data: bytes, run_mode: str, chosen_main: Optional[str]) -> Tuple[int, bytes]:
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + (":" if env.get("ASAN_OPTIONS") else "") + "abort_on_error=1:detect_leaks=0:allocator_may_return_null=1:symbolize=0"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + (":" if env.get("UBSAN_OPTIONS") else "") + "halt_on_error=1:abort_on_error=1:print_stacktrace=0"
        timeout = 0.6

        use_file = False
        if run_mode == "file_or_stdin":
            use_file = True
        elif run_mode == "auto":
            use_file = self._main_likely_expects_file(chosen_main) if chosen_main else False

        if use_file:
            with tempfile.NamedTemporaryFile(prefix="poc_", suffix=".bin", delete=False) as tf:
                tf.write(data)
                tf.flush()
                fname = tf.name
            try:
                p = subprocess.run([bin_path, fname], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, timeout=timeout, env=env)
                return p.returncode, p.stderr or b""
            except subprocess.TimeoutExpired as e:
                return 124, (e.stderr or b"")
            except Exception:
                return 125, b""
            finally:
                try:
                    os.unlink(fname)
                except Exception:
                    pass
        else:
            try:
                p = subprocess.run([bin_path], input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=timeout, env=env)
                return p.returncode, p.stderr or b""
            except subprocess.TimeoutExpired as e:
                return 124, (e.stderr or b"")
            except Exception:
                return 125, b""

    def _main_likely_expects_file(self, main_path: Optional[str]) -> bool:
        if not main_path or not os.path.isfile(main_path):
            return False
        try:
            with open(main_path, "rb") as f:
                d = f.read(300_000).lower()
        except Exception:
            return False
        if b"argv[1]" in d and (b"ifstream" in d or b"fopen" in d or b"open(" in d):
            return True
        if b"argc < 2" in d or b"argc<=1" in d or b"argc <= 1" in d:
            if b"argv[1]" in d:
                return True
        return False

    def _is_target_crash(self, returncode: int, stderr: bytes) -> bool:
        if returncode == 0:
            return False
        low = stderr.lower()
        if b"addresssanitizer" not in low and b"asan" not in low:
            return False
        uaf = b"heap-use-after-free" in low
        df = b"double-free" in low or b"attempting double-free" in low or b"double free" in low
        if not (uaf or df):
            return False
        if b"node" in low and (b"add" in low or b"node::add" in low):
            return True
        return True

    def _has_node_add(self, stderr: bytes) -> bool:
        low = stderr.lower()
        return (b"node::add" in low) or (b"node" in low and b"add" in low)

    def _initial_candidates(self, tokens: List[bytes], prefer_text: bool) -> List[bytes]:
        cands: List[bytes] = []
        cands.append(b"\x00" * 60)
        cands.append(b"A" * 60)
        cands.append(b"0" * 60)
        cands.append(b"\x02\x00\x00\x00" + b"A" * 56)
        cands.append(b"\x02" + b"A" * 59)
        cands.append(b"\x01" * 60)

        words = self._token_words(tokens)
        cmd_words = [w for w in words if w.isalpha() and len(w) <= 10]
        cmd_words_l = [w.lower() for w in cmd_words]
        likely = []
        for w in ("add", "insert", "append", "push", "set", "put", "node", "edge", "link", "child"):
            if w in cmd_words_l:
                likely.append(cmd_words[cmd_words_l.index(w)])
        if not likely and cmd_words:
            likely = cmd_words[:6]

        if likely:
            for w in likely[:10]:
                s = w.encode() + b" a\n" + w.encode() + b" a\n"
                cands.append(s)
                s2 = w.encode() + b" 0 a\n" + w.encode() + b" 0 a\n"
                cands.append(s2)
                s3 = w.encode() + b" a a\n" + w.encode() + b" a a\n"
                cands.append(s3)

        # Common structured text patterns
        cands.append(b"add a a\nadd a a\n")
        cands.append(b"add 0 a\nadd 0 a\n")
        cands.append(b"ADD a a\nADD a a\n")
        cands.append(b"insert a\ninsert a\n")
        cands.append(b"node a\nnode a\n")
        cands.append(b"child a\nchild a\n")
        cands.append(b"push a\npush a\n")

        if prefer_text:
            # Slightly larger sequences to trigger allocation then exception
            cands.append(b"add a\nadd b\nadd a\n")
            cands.append(b"add 0 a\nadd 0 b\nadd 0 a\n")
        return cands

    def _format_guess_candidates(self, tokens: List[bytes]) -> List[bytes]:
        cands: List[bytes] = []
        magics = []
        for t in tokens:
            if 2 <= len(t) <= 8 and all(32 <= b < 127 for b in t):
                if re.fullmatch(rb"[A-Z0-9_]{2,8}", t):
                    magics.append(t)
        magics = magics[:8]
        for m in magics:
            # Try as raw prefix, plus counts and duplicates
            cands.append(m + b"\nadd a\nadd a\n")
            cands.append(m + b"\x02\x00\x00\x00" + b"A" * max(0, 60 - len(m) - 4))
            cands.append(m + b"\x02" + b"A" * max(0, 60 - len(m) - 1))
        # JSON-ish and XML-ish stabs
        cands.append(b'{"a":{"b":1},"a":{"b":2}}')
        cands.append(b'{"cmd":"add","name":"a"}\n{"cmd":"add","name":"a"}\n')
        cands.append(b"<a><b/></a><a><b/></a>")
        return cands

    def _basic_binary_candidates(self, tokens: List[bytes]) -> List[bytes]:
        cands: List[bytes] = []
        # Common little-endian counts at front
        for n in (1, 2, 3, 4, 8, 16):
            cands.append(n.to_bytes(4, "little") + b"A" * 56)
            cands.append(n.to_bytes(2, "little") + b"A" * 58)
            cands.append(bytes([n]) + b"A" * 59)
        # Duplicate records (length-prefixed strings)
        name = b"a"
        rec = bytes([len(name)]) + name
        cands.append(b"\x02" + rec + rec + b"\x00" * (60 - (1 + 2 * len(rec))) if 1 + 2 * len(rec) <= 60 else b"\x02" + rec + rec)
        # Some variation with 32-bit len
        cands.append((2).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"a" + (1).to_bytes(4, "little") + b"a")
        return cands

    def _token_words(self, tokens: List[bytes]) -> List[str]:
        words = []
        for t in tokens:
            if len(t) > 32:
                continue
            try:
                s = t.decode("utf-8", errors="ignore")
            except Exception:
                continue
            s = s.strip()
            if not s:
                continue
            if any(ch in s for ch in "\r\n\t"):
                continue
            if len(s) <= 24:
                words.append(s)
        return words

    def _mutate(self, data: bytes, rng: random.Random, tokens: List[bytes], prefer_text: bool, max_len: int = 256) -> bytes:
        if data is None:
            data = b""
        b = bytearray(data)

        ops = ["flip", "ins", "del", "dup", "setsmall", "tok", "newline"]
        if not prefer_text:
            ops = ["flip", "ins", "del", "dup", "setsmall", "tok"]
        op = rng.choice(ops)

        if op == "flip" and b:
            i = rng.randrange(len(b))
            b[i] ^= 1 << rng.randrange(8)
        elif op == "ins":
            ins_len = rng.randrange(1, 8)
            ins = bytes(rng.randrange(0, 256) for _ in range(ins_len))
            pos = rng.randrange(0, len(b) + 1)
            b[pos:pos] = ins
        elif op == "del" and b:
            if len(b) == 1:
                b = bytearray()
            else:
                a = rng.randrange(0, len(b))
                c = rng.randrange(a + 1, min(len(b), a + 1 + rng.randrange(1, 16)))
                del b[a:c]
        elif op == "dup" and b:
            a = rng.randrange(0, len(b))
            c = rng.randrange(a + 1, min(len(b), a + 1 + rng.randrange(1, 16)))
            seg = b[a:c]
            pos = rng.randrange(0, len(b) + 1)
            b[pos:pos] = seg
        elif op == "setsmall":
            if not b:
                b = bytearray(b"\x02")
            for _ in range(rng.randrange(1, 4)):
                if not b:
                    break
                i = rng.randrange(len(b))
                b[i] = rng.choice([0, 1, 2, 3, 4, 8, 16, 32, 0xFF])
        elif op == "tok" and tokens:
            tok = rng.choice(tokens)
            if len(tok) > 32:
                tok = tok[:32]
            pos = rng.randrange(0, len(b) + 1)
            b[pos:pos] = tok
        elif op == "newline":
            pos = rng.randrange(0, len(b) + 1)
            b[pos:pos] = b"\n"

        if prefer_text and b:
            if rng.random() < 0.35:
                # Encourage duplicate "add" style line patterns
                line = rng.choice([b"add a\nadd a\n", b"add 0 a\nadd 0 a\n", b"insert a\ninsert a\n"])
                pos = rng.randrange(0, len(b) + 1)
                b[pos:pos] = line

        if len(b) > max_len:
            b = b[:max_len]

        return bytes(b)

    def _minimize(self, data: bytes, run_input, start: float, budget: float, require_node_add: bool = True) -> bytes:
        def crashes(d: bytes) -> Tuple[bool, bytes]:
            rc, err = run_input(d)
            ok = self._is_target_crash(rc, err)
            if not ok:
                return False, err
            if require_node_add and not self._has_node_add(err):
                # Prefer the crash that is tied to Node::add, but allow if none.
                return True, err
            return True, err

        ok, err = crashes(data)
        if not ok:
            return data
        have_node_add = self._has_node_add(err)

        cur = data

        # Quick trim from end
        while len(cur) > 1 and time.monotonic() - start < budget * 0.90:
            cand = cur[:-1]
            ok2, err2 = crashes(cand)
            if not ok2:
                break
            if require_node_add and have_node_add and not self._has_node_add(err2):
                break
            cur = cand

        n = len(cur)
        step = max(1, n // 2)
        while step >= 1 and time.monotonic() - start < budget * 0.93:
            changed = False
            i = 0
            while i + step <= len(cur) and time.monotonic() - start < budget * 0.93:
                cand = cur[:i] + cur[i + step:]
                if not cand:
                    i += step
                    continue
                ok2, err2 = crashes(cand)
                if ok2:
                    if require_node_add and have_node_add and not self._has_node_add(err2):
                        i += step
                        continue
                    cur = cand
                    changed = True
                    continue
                i += step
            if not changed:
                step //= 2

        # Byte simplification (limited)
        if time.monotonic() - start < budget * 0.95 and len(cur) <= 256:
            cur_b = bytearray(cur)
            for i in range(len(cur_b)):
                if time.monotonic() - start > budget * 0.96:
                    break
                orig = cur_b[i]
                for v in (0, 1, 2, ord('A'), ord('a'), ord('0'), 0xFF):
                    if v == orig:
                        continue
                    cur_b[i] = v
                    cand = bytes(cur_b)
                    ok2, err2 = crashes(cand)
                    if ok2:
                        if require_node_add and have_node_add and not self._has_node_add(err2):
                            cur_b[i] = orig
                            continue
                        orig = v
                        cur = cand
                        break
                    cur_b[i] = orig

        return cur

    def _trim_trailing(self, data: bytes, run_input, start: float, budget: float) -> bytes:
        cur = data
        while len(cur) > 1 and time.monotonic() - start < budget * 0.985:
            rc, err = run_input(cur[:-1])
            if self._is_target_crash(rc, err):
                cur = cur[:-1]
            else:
                break
        return cur

    def _fallback_payload(self, tokens: List[bytes]) -> bytes:
        # Try to construct a plausible duplicate-add text payload using any discovered token
        words = self._token_words(tokens)
        for w in words:
            wl = w.lower()
            if wl in ("add", "insert", "append", "push", "put", "set"):
                return (w.encode() + b" a\n" + w.encode() + b" a\n")
        return b"add a\nadd a\n"


def shutil_which(cmd: str) -> Optional[str]:
    if os.path.isabs(cmd) and os.access(cmd, os.X_OK):
        return cmd
    path = os.environ.get("PATH", "")
    exts = [""]
    if os.name == "nt":
        pathext = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
        exts = pathext
    for d in path.split(os.pathsep):
        d = d.strip('"')
        if not d:
            continue
        for e in exts:
            p = os.path.join(d, cmd + e)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    return None