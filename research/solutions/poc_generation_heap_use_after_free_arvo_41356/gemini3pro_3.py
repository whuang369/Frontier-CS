import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import glob
import random
import time
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a temporary workspace
        work_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        
        try:
            # Extract source code
            with tarfile.open(src_path) as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory
                
                safe_members = []
                for member in tar.getmembers():
                    if is_within_directory(work_dir, os.path.join(work_dir, member.name)):
                        safe_members.append(member)
                tar.extractall(path=work_dir, members=safe_members)

            # Identify source root
            src_root = work_dir
            contents = os.listdir(work_dir)
            if len(contents) == 1 and os.path.isdir(os.path.join(work_dir, contents[0])):
                src_root = os.path.join(work_dir, contents[0])

            # Compile with ASAN
            binary = self._compile(src_root)
            if not binary:
                return b""

            # Gather seeds from source tests/docs
            seeds = self._gather_seeds(src_root)
            if not seeds:
                # Fallback seeds covering common formats (XML, JSON, YAML)
                seeds = [
                    b"<root></root>", b"{}", b"[]", b"key: value", 
                    b"A" * 10, b"\x00" * 4, b"&a [*a]"
                ]

            # Run Fuzzing
            crash_input = self._fuzz(binary, seeds)
            
            # Minimize if crash found
            if crash_input:
                crash_input = self._minimize(binary, crash_input)
                return crash_input
            
            return b""

        finally:
            os.chdir(old_cwd)
            shutil.rmtree(work_dir, ignore_errors=True)

    def _compile(self, src_root):
        env = os.environ.copy()
        # Ensure AddressSanitizer is enabled
        flags = "-fsanitize=address -g -O1"
        env['CFLAGS'] = flags
        env['CXXFLAGS'] = flags
        env['LDFLAGS'] = flags
        
        # Prefer clang if available
        try:
            subprocess.run(["clang", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
        except FileNotFoundError:
            env['CC'] = 'gcc'
            env['CXX'] = 'g++'

        # Build strategy: CMake -> Configure -> Make
        build_methods = [
            (self._build_cmake, os.path.join(src_root, "CMakeLists.txt")),
            (self._build_configure, os.path.join(src_root, "configure")),
            (self._build_make, os.path.join(src_root, "Makefile"))
        ]

        for build_func, indicator in build_methods:
            if os.path.exists(indicator):
                if build_func(src_root, env):
                    binary = self._find_binary(src_root)
                    if binary: return binary
        
        # Last ditch: try to find an existing binary if build failed silently or wasn't needed
        return self._find_binary(src_root)

    def _build_cmake(self, root, env):
        build_dir = os.path.join(root, "build_fuzz")
        os.makedirs(build_dir, exist_ok=True)
        try:
            subprocess.run(["cmake", ".."], cwd=build_dir, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["make", f"-j{os.cpu_count()}"], cwd=build_dir, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _build_configure(self, root, env):
        try:
            subprocess.run(["chmod", "+x", "./configure"], cwd=root, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["./configure"], cwd=root, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["make", f"-j{os.cpu_count()}"], cwd=root, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _build_make(self, root, env):
        try:
            subprocess.run(["make", f"-j{os.cpu_count()}"], cwd=root, env=env, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _find_binary(self, root):
        candidates = []
        for r, d, f in os.walk(root):
            for file in f:
                path = os.path.join(r, file)
                if os.access(path, os.X_OK) and not file.endswith('.sh') and not file.endswith('.py'):
                    try:
                        with open(path, "rb") as bf:
                            if bf.read(4) == b"\x7fELF":
                                candidates.append(path)
                    except:
                        pass
        
        if not candidates:
            return None
        # Return the largest binary, assuming it contains the main logic/symbols
        candidates.sort(key=lambda x: os.path.getsize(x), reverse=True)
        return candidates[0]

    def _gather_seeds(self, root):
        seeds = []
        exts = {'.xml', '.json', '.yaml', '.yml', '.txt'}
        for r, d, f in os.walk(root):
            for file in f:
                if any(file.endswith(e) for e in exts) or 'test' in file.lower():
                    path = os.path.join(r, file)
                    if os.path.getsize(path) < 2048:
                        try:
                            with open(path, "rb") as f_in:
                                seeds.append(f_in.read())
                        except:
                            pass
        return seeds if seeds else None

    def _check_crash(self, binary, data):
        # Writes data to a file and runs the binary against it
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data)
            tf_name = tf.name
        
        crashed = False
        try:
            # Try file argument
            res = subprocess.run([binary, tf_name], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1.5)
            if res.returncode != 0 and b"AddressSanitizer" in res.stderr:
                crashed = True
            elif res.returncode != 0:
                # If non-ASAN crash, try stdin just in case
                res = subprocess.run([binary], input=data, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1.5)
                if res.returncode != 0 and b"AddressSanitizer" in res.stderr:
                    crashed = True
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        finally:
            if os.path.exists(tf_name):
                os.unlink(tf_name)
        return crashed

    def _fuzz(self, binary, seeds):
        # Try initial seeds first
        for s in seeds:
            if self._check_crash(binary, s):
                return s

        start_time = time.time()
        timeout = 45 # Time limit for fuzzing
        found_crash = None

        def worker():
            nonlocal found_crash
            while time.time() - start_time < timeout and found_crash is None:
                seed = random.choice(seeds)
                mutated = self._mutate(seed)
                if self._check_crash(binary, mutated):
                    found_crash = mutated
                    return

        with ThreadPoolExecutor(max_workers=8) as exc:
            futures = [exc.submit(worker) for _ in range(8)]
            for f in futures:
                f.result()
                if found_crash:
                    break
        
        return found_crash

    def _mutate(self, data):
        if not data: return b"A"
        arr = bytearray(data)
        op = random.randint(0, 5)
        
        if op == 0 and len(arr) > 0: # Bit flip
            idx = random.randint(0, len(arr)-1)
            arr[idx] ^= (1 << random.randint(0, 7))
        elif op == 1 and len(arr) > 0: # Byte flip
            idx = random.randint(0, len(arr)-1)
            arr[idx] = random.randint(0, 255)
        elif op == 2: # Insert magic
            magic = [b"\xFF", b"\x00", b"\x7F", b"%n", b"\n"]
            pos = random.randint(0, len(arr))
            for b in random.choice(magic):
                arr.insert(pos, b)
        elif op == 3: # Delete
            if len(arr) > 1:
                l = random.randint(1, min(len(arr), 10))
                s = random.randint(0, len(arr)-l)
                del arr[s:s+l]
        elif op == 4: # Duplicate/Swap
            if len(arr) > 2:
                p1 = random.randint(0, len(arr)-2)
                l = random.randint(1, min(len(arr)-p1, 10))
                chunk = arr[p1:p1+l]
                p2 = random.randint(0, len(arr))
                arr[p2:p2] = chunk
        elif op == 5: # Structural Injection
            chars = b"{}[]<>&*!|:,\"'\\"
            pos = random.randint(0, len(arr))
            arr.insert(pos, random.choice(chars))
            
        return bytes(arr)

    def _minimize(self, binary, data):
        current = data
        changed = True
        while changed:
            changed = False
            # Remove chunk
            chunk_size = max(1, len(current) // 20)
            for i in range(0, len(current) - chunk_size + 1):
                candidate = current[:i] + current[i+chunk_size:]
                if self._check_crash(binary, candidate):
                    current = candidate
                    changed = True
                    break
            if not changed:
                # Remove single byte
                for i in range(len(current)):
                    candidate = current[:i] + current[i+1:]
                    if self._check_crash(binary, candidate):
                        current = candidate
                        changed = True
                        break
        return current