import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import glob
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        # Create temporary workspace
        base_tmp = tempfile.mkdtemp()
        try:
            # Extract source
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path) as tar:
                    tar.extractall(base_tmp)
            
            # Locate actual source root
            src_root = base_tmp
            contents = os.listdir(base_tmp)
            if len(contents) == 1 and os.path.isdir(os.path.join(base_tmp, contents[0])):
                src_root = os.path.join(base_tmp, contents[0])

            # Build the project
            exe_path = self.build_project(src_root)
            
            if not exe_path:
                # Fallback if build fails
                return b'A' * 60

            # Collect seeds from the source tree
            seeds = self.collect_seeds(src_root)
            
            # Fuzz the executable to find the crash
            poc = self.fuzz(exe_path, seeds)
            return poc
        finally:
            shutil.rmtree(base_tmp, ignore_errors=True)

    def build_project(self, src_root):
        env = os.environ.copy()
        # Enable ASAN to detect UAF/Double-Free
        # -O1 is often good for ASAN stack traces, -g for symbols
        flags = "-fsanitize=address -g -O1"
        env['CFLAGS'] = flags
        env['CXXFLAGS'] = flags
        
        # Strategy 1: CMake
        if os.path.exists(os.path.join(src_root, 'CMakeLists.txt')):
            build_dir = os.path.join(src_root, 'build_fuzz')
            os.makedirs(build_dir, exist_ok=True)
            try:
                subprocess.run(['cmake', '..'], cwd=build_dir, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                subprocess.run(['make', '-j8'], cwd=build_dir, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                return self.find_executable(build_dir)
            except:
                pass

        # Strategy 2: Configure/Make
        if os.path.exists(os.path.join(src_root, 'configure')):
            try:
                subprocess.run(['./configure'], cwd=src_root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                subprocess.run(['make', '-j8'], cwd=src_root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                return self.find_executable(src_root)
            except:
                pass
        
        # Strategy 3: Makefile
        if os.path.exists(os.path.join(src_root, 'Makefile')):
            try:
                subprocess.run(['make', '-j8'], cwd=src_root, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                return self.find_executable(src_root)
            except:
                pass

        # Strategy 4: Direct Compilation (Fallback)
        # Attempt to compile all C++ files found
        cpp_files = []
        for root, dirs, files in os.walk(src_root):
            for f in files:
                if f.endswith(('.cpp', '.cc', '.cxx')):
                    cpp_files.append(os.path.join(root, f))
        
        if cpp_files:
            out_bin = os.path.join(src_root, 'vuln_target')
            # Look for a file containing main
            main_file = None
            for f in cpp_files:
                try:
                    with open(f, 'r', errors='ignore') as fd:
                        if 'int main' in fd.read():
                            main_file = f
                            break
                except: pass
            
            if main_file:
                # Compile everything (naive)
                cmd = ['g++', '-fsanitize=address', '-g'] + cpp_files + ['-o', out_bin]
                try:
                    subprocess.run(cmd, cwd=src_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                    if os.path.exists(out_bin):
                        return out_bin
                except:
                    pass

        return None

    def find_executable(self, search_dir):
        candidates = []
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                path = os.path.join(root, f)
                if os.access(path, os.X_OK) and not os.path.isdir(path):
                    if f.endswith(('.sh', '.py', '.pl', '.so', '.a', '.o')): continue
                    candidates.append(path)
        
        # Heuristics to pick the right executable
        # Priority: explicit 'arvo' or 'fuzz' or 'test'
        for c in candidates:
            base = os.path.basename(c).lower()
            if 'arvo' in base: return c
        for c in candidates:
            base = os.path.basename(c).lower()
            if 'fuzz' in base: return c
        for c in candidates:
            base = os.path.basename(c).lower()
            if 'test' in base: return c
            
        if candidates:
            return candidates[0]
        return None

    def collect_seeds(self, src_root):
        seeds = []
        extensions = {'.json', '.xml', '.dat', '.txt', '.avro'}
        for root, dirs, files in os.walk(src_root):
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in extensions or 'test' in f or 'sample' in f:
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'rb') as fd:
                            content = fd.read()
                            if 0 < len(content) < 5000:
                                seeds.append(content)
                    except: pass
        
        if not seeds:
            seeds = [b'A'*10, b'{ "key": "value" }', b'\x00'*20]
        return seeds

    def fuzz(self, executable, seeds):
        start_time = time.time()
        # Cap fuzzing time
        end_time = start_time + 45
        
        corpus = list(seeds)
        
        # Setup run environment with library paths
        env = os.environ.copy()
        exe_dir = os.path.dirname(executable)
        libs = [exe_dir, os.path.join(exe_dir, '../lib'), os.path.join(exe_dir, 'lib')]
        env['LD_LIBRARY_PATH'] = ':'.join(libs) + ':' + env.get('LD_LIBRARY_PATH', '')
        
        while time.time() < end_time:
            seed = random.choice(corpus)
            
            # Mutation
            if not seed:
                data = bytearray(b'A')
            else:
                data = bytearray(seed)
                
            mutations = random.randint(1, 3)
            for _ in range(mutations):
                if not data: break
                op = random.randint(0, 3)
                idx = random.randint(0, len(data)-1)
                if op == 0: # Flip bit
                    data[idx] ^= (1 << random.randint(0,7))
                elif op == 1: # Byte replace
                    data[idx] = random.randint(0, 255)
                elif op == 2: # Delete
                    del data[idx]
                elif op == 3: # Insert
                    data.insert(idx, random.randint(0, 255))
            
            payload = bytes(data)
            
            # Try running via Stdin
            try:
                proc = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                _, stderr = proc.communicate(input=payload, timeout=0.2)
                if proc.returncode != 0 and (b"AddressSanitizer" in stderr or b"double-free" in stderr):
                    return payload
            except:
                pass
                
            # Try running via File argument
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(payload)
                    tf_name = tf.name
                
                try:
                    proc = subprocess.Popen([executable, tf_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    _, stderr = proc.communicate(timeout=0.2)
                    if proc.returncode != 0 and (b"AddressSanitizer" in stderr or b"double-free" in stderr):
                        return payload
                finally:
                    if os.path.exists(tf_name): os.remove(tf_name)
            except:
                pass
        
        # Return something if no crash found
        seeds.sort(key=lambda x: abs(len(x)-60))
        return seeds[0]