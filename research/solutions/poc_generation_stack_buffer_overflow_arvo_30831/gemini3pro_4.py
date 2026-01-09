import os
import sys
import subprocess
import tempfile
import random
import time
import shutil
import re
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability by compiling the source with ASAN and fuzzing it.
        """
        # Create a temporary directory for building and fuzzing
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = os.path.join(temp_dir, "build")
            # Copy source to build_dir to avoid modifying original and allow write access
            try:
                shutil.copytree(src_path, build_dir, dirs_exist_ok=True)
            except Exception:
                # If copy fails, we might be unable to proceed, return a fallback
                return b"\x40\x01\x12\x34"

            c_files = []
            cpp_files = []
            harness_file = None
            files_with_main = []

            # 1. Identify source files and harness
            for root, _, files in os.walk(build_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    if f.endswith(".c"):
                        c_files.append(full_path)
                    elif f.endswith(".cc") or f.endswith(".cpp") or f.endswith(".cxx"):
                        cpp_files.append(full_path)
                    
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as fd:
                            content = fd.read()
                            # Check for libFuzzer entry point
                            if "LLVMFuzzerTestOneInput" in content:
                                harness_file = full_path
                            # Check for main function to exclude from linking (avoid duplicates)
                            # Regex matches 'int main(' or 'void main(' with optional spaces
                            if re.search(r'\b(int|void)\s+main\s*\(', content):
                                files_with_main.append(full_path)
                    except Exception:
                        pass
            
            # If no harness is found, we can't fuzz. Return a generic CoAP packet.
            if not harness_file:
                return b"\x40\x01\x12\x34\x00"

            # 2. Setup Compiler and Driver
            is_cpp = (len(cpp_files) > 0) or harness_file.endswith((".cc", ".cpp", ".cxx"))
            compiler = "clang++" if is_cpp else "clang"
            
            driver_path = os.path.join(build_dir, "fuzz_driver.cpp" if is_cpp else "fuzz_driver.c")
            
            # Driver code to wrap LLVMFuzzerTestOneInput with a main that reads from file
            driver_code = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
#else
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
#endif

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(len);
    if (!buf) { fclose(f); return 0; }
    fread(buf, 1, len, f);
    fclose(f);
    LLVMFuzzerTestOneInput(buf, len);
    free(buf);
    return 0;
}
"""
            with open(driver_path, "w") as f:
                f.write(driver_code)
            
            # 3. Compile
            bin_path = os.path.join(build_dir, "fuzzer_bin")
            
            # Construct list of files to compile
            # Include driver, harness, and all other sources NOT containing main
            srcs = [driver_path]
            for f in c_files + cpp_files:
                if f == harness_file:
                    srcs.append(f)
                elif f not in files_with_main:
                    srcs.append(f)
            
            # Compile command with ASAN
            cmd = [compiler, "-g", "-O1", "-fsanitize=address", "-o", bin_path] + srcs + [f"-I{build_dir}"]
            
            # Attempt compilation
            compile_res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=build_dir)
            
            if compile_res.returncode != 0 or not os.path.exists(bin_path):
                # Fallback: return a guessed packet if compilation fails
                return b"\x40\x01\x12\x34"

            # 4. Fuzzing Loop
            start_time = time.time()
            # Try to fuzz for up to 45 seconds (leaving buffer for setup/teardown)
            time_limit = 45 
            
            # Initial corpus: CoAP structures
            # Header(4) + Token(0-8) + Options + Payload
            corpus = [
                b"\x40\x01\x12\x34", # Empty GET
                b"\x40\x00\x00\x00", # Empty
                b"\x60\x44\x12\x34", # ACK
                b"\x50\x02\x11\x11", # NON POST
            ]
            
            best_crash = None

            while time.time() - start_time < time_limit:
                # Generation / Mutation
                if corpus and random.random() < 0.7:
                    base = random.choice(corpus)
                    data = bytearray(base)
                    # Simple mutations
                    mutations = random.randint(1, 3)
                    for _ in range(mutations):
                        op = random.randint(0, 3)
                        if op == 0 and len(data) > 0: # Flip
                            data[random.randint(0, len(data)-1)] ^= random.randint(1, 255)
                        elif op == 1: # Insert
                            data.insert(random.randint(0, len(data)), random.randint(0, 255))
                        elif op == 2 and len(data) > 0: # Delete
                            del data[random.randint(0, len(data)-1)]
                        elif op == 3: # Append
                            data.append(random.randint(0, 255))
                    inp = bytes(data)
                else:
                    # Generate fresh CoAP-like packet
                    ver = 1
                    t = random.randint(0, 3)
                    tkl = random.randint(0, 8)
                    code = random.randint(0, 255)
                    mid = random.randint(0, 65535)
                    pkt = bytearray([(ver<<6)|(t<<4)|tkl, code, mid>>8, mid&0xff])
                    pkt.extend(os.urandom(tkl))
                    # Add random options
                    for _ in range(random.randint(0, 5)):
                        # Option delta/length byte
                        pkt.append(random.randint(0, 255))
                        # Option value (random length, small for speed)
                        pkt.extend(os.urandom(random.randint(0, 4)))
                    inp = bytes(pkt)
                
                # Truncate to reasonable size (ground truth is 21 bytes)
                if len(inp) > 64:
                    inp = inp[:64]

                # Run Harness
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(inp)
                    tf_name = tf.name
                
                try:
                    # Run with timeout
                    proc = subprocess.run([bin_path, tf_name], capture_output=True, timeout=0.5)
                    
                    # Check for crash
                    if proc.returncode != 0:
                        # Look for ASAN report or Segfault
                        if b"AddressSanitizer" in proc.stderr or b"segmentation fault" in proc.stderr.lower():
                            # Found a crash!
                            best_crash = inp
                            break # Stop early on crash
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
                finally:
                    if os.path.exists(tf_name):
                        os.remove(tf_name)
                
                # Optional: Feedback loop - if new path (not easily detectable here), add to corpus.
                # Just add valid-looking small inputs to corpus occasionally
                if len(inp) <= 30 and random.random() < 0.1:
                    corpus.append(inp)

            if best_crash:
                return best_crash
            
            # If no crash found, return a likely candidate (valid header + some option bytes)
            return b"\x40\x01\x12\x34\xd1\x01\x01"