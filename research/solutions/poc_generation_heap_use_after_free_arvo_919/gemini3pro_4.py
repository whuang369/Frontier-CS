import os
import sys
import tarfile
import subprocess
import tempfile
import shutil
import random
import time
import struct
import base64
from concurrent.futures import ThreadPoolExecutor

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        crash_result = []

        try:
            # 1. Extract source
            with tarfile.open(src_path) as tar:
                tar.extractall(path=work_dir)
            
            src_root = work_dir
            for root, dirs, files in os.walk(work_dir):
                if 'configure.ac' in files or 'meson.build' in files or 'autogen.sh' in files:
                    src_root = root
                    break
            
            # 2. Build with ASAN
            env = os.environ.copy()
            env['CC'] = 'clang'
            env['CXX'] = 'clang++'
            env['CFLAGS'] = '-fsanitize=address -g'
            env['CXXFLAGS'] = '-fsanitize=address -g'
            env['LDFLAGS'] = '-fsanitize=address'
            
            binary_path = os.path.join(src_root, 'ots-sanitize')
            build_success = False
            
            # Try autogen
            if os.path.exists(os.path.join(src_root, 'autogen.sh')):
                subprocess.run(['./autogen.sh'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Try configure and make
            if os.path.exists(os.path.join(src_root, 'configure')):
                subprocess.run(['./configure'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['make', '-j8'], cwd=src_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(binary_path):
                    build_success = True
            
            if not build_success:
                # Search for binary if build system structure was different or pre-built
                for root, dirs, files in os.walk(work_dir):
                    if 'ots-sanitize' in files:
                        binary_path = os.path.join(root, 'ots-sanitize')
                        os.chmod(binary_path, 0o755)
                        build_success = True
                        break

            # 3. Collect Seeds
            seeds = []
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    if f.lower().endswith(('.ttf', '.otf', '.woff', '.woff2')):
                        try:
                            with open(os.path.join(root, f), 'rb') as fd:
                                content = fd.read()
                                if len(content) < 200 * 1024: 
                                    seeds.append(content)
                        except:
                            pass
            
            # Fallback seed (Minimal TTF)
            fallback_seed = base64.b64decode("AAEAAAAKAIAAAwAgT1MvMgAAAAAAAACsAAAAWGNtYXAA3AAAAAAAAABnbHlmAAAAAAAAAFgAAAAAAAAAaGVhZAAAAAEQAAAAOGhoZWEAAAFkAAAAJGhtdHgAAAcQAAAAUbG9jYQAAAAG4AAAADG1heHAAAAABzAAAACBuYW1lAAAB3AAAACBwb3N0AAAB7AAAACAAAgAAAAEAAQAAAAAAAAABAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAACAAAABAAABQYAAAAAAAAAAAABAAA=")
            if not seeds:
                seeds.append(fallback_seed)
            else:
                seeds.sort(key=len)
                seeds = seeds[:30]

            # 4. Fuzzing
            start_time = time.time()
            # Stop if we find a crash or run out of time (5 minutes)
            
            def fuzz_worker():
                local_rnd = random.Random()
                while time.time() - start_time < 300:
                    if crash_result:
                        return
                    
                    seed = local_rnd.choice(seeds)
                    mutated = bytearray(seed)
                    
                    # Mutate
                    num_mutations = local_rnd.randint(1, 8)
                    for _ in range(num_mutations):
                        if not mutated: break
                        op = local_rnd.randint(0, 4)
                        if op == 0: # Flip byte
                            idx = local_rnd.randint(0, len(mutated) - 1)
                            mutated[idx] ^= 0xFF
                        elif op == 1: # Flip bit
                            idx = local_rnd.randint(0, len(mutated) - 1)
                            bit = local_rnd.randint(0, 7)
                            mutated[idx] ^= (1 << bit)
                        elif op == 2: # Delete chunk
                            if len(mutated) > 5:
                                l = local_rnd.randint(1, 8)
                                idx = local_rnd.randint(0, len(mutated) - l)
                                del mutated[idx:idx+l]
                        elif op == 3: # Insert junk
                            l = local_rnd.randint(1, 8)
                            idx = local_rnd.randint(0, len(mutated))
                            mutated[idx:idx] = os.urandom(l)
                        elif op == 4: # Overwrite int
                            if len(mutated) > 4:
                                idx = local_rnd.randint(0, len(mutated) - 4)
                                val = struct.pack("<I", local_rnd.randint(0, 0xFFFFFFFF))
                                mutated[idx:idx+4] = val
                    
                    # Test
                    tf_name = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False) as tf:
                            tf.write(mutated)
                            tf_name = tf.name
                        
                        proc = subprocess.run(
                            [binary_path, tf_name], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.PIPE,
                            timeout=1
                        )
                        
                        if proc.returncode != 0:
                            err = proc.stderr
                            if b"AddressSanitizer" in err and b"heap-use-after-free" in err:
                                crash_result.append(bytes(mutated))
                                return
                    except:
                        pass
                    finally:
                        if tf_name and os.path.exists(tf_name):
                            os.remove(tf_name)

            if build_success:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(fuzz_worker) for _ in range(8)]
                    for f in futures:
                        f.result()
                        if crash_result:
                            break
            
            if crash_result:
                return crash_result[0]
            
            return fallback_seed

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)