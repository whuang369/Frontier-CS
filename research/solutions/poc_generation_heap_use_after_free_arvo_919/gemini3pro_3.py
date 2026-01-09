import os
import sys
import subprocess
import tempfile
import shutil
import glob
import random
import time
import concurrent.futures

class Solution:
    def solve(self, src_path: str) -> bytes:
        base_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            extract_dir = os.path.join(base_dir, "source")
            os.makedirs(extract_dir, exist_ok=True)
            try:
                shutil.unpack_archive(src_path, extract_dir)
            except Exception:
                return b""

            # Locate root directory (handle cases where tarball contains a single top-level folder)
            root_dir = extract_dir
            entries = os.listdir(extract_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
                root_dir = os.path.join(extract_dir, entries[0])

            # 2. Build the project with AddressSanitizer (ASAN)
            env = os.environ.copy()
            flags = "-fsanitize=address -g -O1"
            env["CFLAGS"] = flags
            env["CXXFLAGS"] = flags
            env["LDFLAGS"] = flags
            env["CC"] = "clang"
            env["CXX"] = "clang++"

            executable = None

            # Strategy A: Autotools (./autogen.sh && ./configure && make)
            if os.path.exists(os.path.join(root_dir, "autogen.sh")):
                subprocess.run(["./autogen.sh"], cwd=root_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["./configure", "--enable-debug", "--enable-werror=no"], cwd=root_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["make", "-j8"], cwd=root_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                possible_exes = [
                    os.path.join(root_dir, "util", "ots-sanitize"),
                    os.path.join(root_dir, "ots-sanitize")
                ]
                for p in possible_exes:
                    if os.path.exists(p):
                        executable = p
                        break
            
            # Strategy B: Meson (meson setup build && ninja)
            if not executable and os.path.exists(os.path.join(root_dir, "meson.build")):
                build_dir = os.path.join(root_dir, "build_meson")
                try:
                    subprocess.run(["meson", "setup", build_dir, "-Db_sanitize=address"], cwd=root_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["ninja", "-C", build_dir], cwd=root_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    possible_exes = [
                        os.path.join(build_dir, "util", "ots-sanitize"),
                        os.path.join(build_dir, "ots-sanitize")
                    ]
                    for p in possible_exes:
                        if os.path.exists(p):
                            executable = p
                            break
                except FileNotFoundError:
                    pass

            # If build failed, we cannot generate a verified PoC
            if not executable:
                return b""

            # 3. Collect Seeds for Fuzzing
            # OTS source usually contains test fonts in 'tests' or 'data' directories
            seed_files = []
            for ext in ["ttf", "otf", "woff", "woff2"]:
                seed_files.extend(glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True))
            
            seeds = []
            for sf in seed_files:
                try:
                    if os.path.getsize(sf) < 50 * 1024:  # Use small files to keep throughput high
                        with open(sf, "rb") as f:
                            seeds.append(f.read())
                except:
                    pass
            
            if not seeds:
                # Fallback minimal seed (TTF header-like)
                seeds = [b'\x00\x01\x00\x00' + b'\x00' * 20]
            else:
                # Limit seeds to a manageable number
                seeds = random.sample(seeds, min(len(seeds), 50))

            # 4. Fuzzer Worker Function
            def fuzz_worker(worker_idx, seeds, exe, duration):
                start_time = time.time()
                while time.time() - start_time < duration:
                    # Pick a seed and mutate it
                    seed = random.choice(seeds)
                    mutated = bytearray(seed)
                    
                    # Mutation strategies
                    mutation_type = random.random()
                    if mutation_type < 0.5:
                        # Bit flips
                        num_flips = random.randint(1, 4)
                        for _ in range(num_flips):
                            idx = random.randint(0, len(mutated) - 1)
                            mutated[idx] ^= (1 << random.randint(0, 7))
                    elif mutation_type < 0.8:
                        # Byte overwrites
                        idx = random.randint(0, len(mutated) - 1)
                        mutated[idx] = random.randint(0, 255)
                    else:
                        # Chunk manipulation (delete or insert)
                        if len(mutated) > 5:
                            chunk_len = random.randint(1, 4)
                            chunk_pos = random.randint(0, len(mutated) - chunk_len)
                            if random.random() < 0.5:
                                del mutated[chunk_pos:chunk_pos+chunk_len]
                            else:
                                mutated[chunk_pos:chunk_pos] = os.urandom(chunk_len)

                    if not mutated:
                        continue

                    # Write to temporary file
                    test_file = os.path.join(tempfile.gettempdir(), f"poc_{worker_idx}_{random.randint(0, 1000000)}.font")
                    with open(test_file, "wb") as f:
                        f.write(mutated)

                    try:
                        # Run the vulnerable binary
                        proc = subprocess.run(
                            [exe, test_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=2
                        )
                        
                        # Check for crash (ASAN output)
                        if proc.returncode != 0:
                            stderr_out = proc.stderr.decode(errors='replace')
                            if "AddressSanitizer" in stderr_out and "heap-use-after-free" in stderr_out:
                                with open(test_file, "rb") as f:
                                    crash_bytes = f.read()
                                os.unlink(test_file)
                                return crash_bytes
                    except subprocess.TimeoutExpired:
                        pass
                    finally:
                        if os.path.exists(test_file):
                            os.unlink(test_file)
                            
                return None

            # 5. Run Fuzzing in Parallel
            num_workers = 8
            fuzz_duration = 60 # seconds

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(fuzz_worker, i, seeds, executable, fuzz_duration) for i in range(num_workers)]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        # Crash found
                        return result
            
            # If no crash found within time limit
            return b""

        except Exception:
            return b""
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)