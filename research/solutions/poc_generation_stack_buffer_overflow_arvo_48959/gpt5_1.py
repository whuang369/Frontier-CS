import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # GZIP header: ID1, ID2, CM=8 (deflate), FLG=0, MTIME=0, XFL=0, OS=3 (Unix)
        header = bytes([0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03])
        # Deflate block: BFINAL=1, BTYPE=01 (fixed Huffman), followed by EOB (256) -> 7 zero bits
        deflate = bytes([0x03, 0x00])
        # Trailer: CRC32=0, ISIZE=0 for empty uncompressed data
        trailer = struct.pack("<II", 0, 0)
        return header + deflate + trailer