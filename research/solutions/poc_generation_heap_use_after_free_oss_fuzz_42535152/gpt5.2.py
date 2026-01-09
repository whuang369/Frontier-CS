import struct
from typing import Dict, Tuple, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        num_dups = 100
        first_dup_obj = 10
        last_dup_obj = first_dup_obj + num_dups - 1
        size = max(110, last_dup_obj + 1)  # ensure at least covers 0..last_dup_obj

        def make_indirect_obj(objnum: int, body: bytes) -> bytes:
            return (f"{objnum} 0 obj\n".encode("ascii") + body + b"\nendobj\n")

        def make_stream_obj(objnum: int, dict_body: bytes, stream_data: bytes) -> bytes:
            return (
                f"{objnum} 0 obj\n".encode("ascii")
                + b"<< "
                + dict_body
                + b" >>\nstream\n"
                + stream_data
                + b"\nendstream\nendobj\n"
            )

        def build_objstm(objstm_objnum: int, objnums: List[int], rev: int) -> bytes:
            data_parts: List[bytes] = []
            offsets: Dict[int, int] = {}
            off = 0

            kids_list = b" ".join([f"{i} 0 R".encode("ascii") for i in range(first_dup_obj + 1, last_dup_obj + 1)])
            for on in objnums:
                if on == first_dup_obj:
                    body = (
                        b"<< /Type /Node /Rev "
                        + str(rev).encode("ascii")
                        + b" /Kids ["
                        + kids_list
                        + b"] >>\n"
                    )
                else:
                    body = (
                        b"<< /I "
                        + str(on).encode("ascii")
                        + b" /Rev "
                        + str(rev).encode("ascii")
                        + b" >>\n"
                    )
                offsets[on] = off
                data_parts.append(body)
                off += len(body)

            header = (" ".join([f"{on} {offsets[on]}" for on in objnums]) + "\n").encode("ascii")
            first = len(header)
            stream_data = header + b"".join(data_parts)

            dict_body = (
                b"/Type /ObjStm /N "
                + str(len(objnums)).encode("ascii")
                + b" /First "
                + str(first).encode("ascii")
                + b" /Length "
                + str(len(stream_data)).encode("ascii")
            )
            return make_stream_obj(objstm_objnum, dict_body, stream_data)

        def pack_xref_entry(entry_type: int, f2: int, f3: int) -> bytes:
            return struct.pack(">B", entry_type) + struct.pack(">I", f2 & 0xFFFFFFFF) + struct.pack(">H", f3 & 0xFFFF)

        def build_xref_stream_obj(
            xref_objnum: int,
            root_objnum: int,
            size_val: int,
            offsets_type1: Dict[int, int],
            compressed_map: Dict[int, Tuple[int, int]],
            prev: int | None,
            xref_obj_offset: int,
        ) -> bytes:
            entries = [pack_xref_entry(0, 0, 65535)]
            for objn in range(1, size_val):
                if objn in compressed_map:
                    stmn, idx = compressed_map[objn]
                    entries.append(pack_xref_entry(2, stmn, idx))
                elif objn in offsets_type1:
                    entries.append(pack_xref_entry(1, offsets_type1[objn], 0))
                elif objn == xref_objnum:
                    entries.append(pack_xref_entry(1, xref_obj_offset, 0))
                else:
                    entries.append(pack_xref_entry(0, 0, 65535))

            xref_data = b"".join(entries)
            dict_parts = [
                b"/Type /XRef",
                b"/Size " + str(size_val).encode("ascii"),
                b"/W [1 4 2]",
                b"/Root " + str(root_objnum).encode("ascii") + b" 0 R",
                b"/Length " + str(len(xref_data)).encode("ascii"),
            ]
            if prev is not None:
                dict_parts.append(b"/Prev " + str(prev).encode("ascii"))
            dict_body = b" ".join(dict_parts)
            return make_stream_obj(xref_objnum, dict_body, xref_data)

        # Fixed objects
        header = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n"

        catalog = (
            b"<< /Type /Catalog /Pages 2 0 R "
            + b"/StructTreeRoot "
            + str(first_dup_obj).encode("ascii")
            + b" 0 R "
            + b"/OldObjStm 6 0 R /NewObjStm 7 0 R >>"
        )
        pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << >> >>"
        contents_data = b"q\nQ\n"
        contents = b"<< /Length " + str(len(contents_data)).encode("ascii") + b" >>"

        # Base revision objects
        buf = bytearray()
        buf += header

        offsets_base: Dict[int, int] = {}
        for objnum, body, is_stream, sdata in [
            (1, catalog, False, b""),
            (2, pages, False, b""),
            (3, page, False, b""),
            (4, contents, True, contents_data),
        ]:
            offsets_base[objnum] = len(buf)
            if is_stream:
                buf += make_stream_obj(objnum, body[3:-3] if body.startswith(b"<<") and body.endswith(b">>") else body, sdata)  # not used
            else:
                buf += make_indirect_obj(objnum, body)

        # Correctly create stream object 4
        # Replace the previously appended incorrect object 4 (if any) by rebuilding buffer cleanly
        buf = bytearray()
        buf += header
        offsets_base = {}

        offsets_base[1] = len(buf)
        buf += make_indirect_obj(1, catalog)
        offsets_base[2] = len(buf)
        buf += make_indirect_obj(2, pages)
        offsets_base[3] = len(buf)
        buf += make_indirect_obj(3, page)
        offsets_base[4] = len(buf)
        buf += make_stream_obj(4, b"/Length " + str(len(contents_data)).encode("ascii"), contents_data)

        # Object stream in base revision (6 0)
        objnums = list(range(first_dup_obj, last_dup_obj + 1))
        offsets_base[6] = len(buf)
        buf += build_objstm(6, objnums, rev=0)

        # Base xref stream object (8 0)
        base_xref_objnum = 8
        base_xref_offset = len(buf)

        compressed_base: Dict[int, Tuple[int, int]] = {}
        for i, on in enumerate(objnums):
            compressed_base[on] = (6, i)

        # Base offsets_type1 includes objects present in base excluding xref (added by builder)
        offsets_type1_base = dict(offsets_base)
        # Ensure object 7 and 9 are free in base (not in offsets_type1_base)

        buf += build_xref_stream_obj(
            xref_objnum=base_xref_objnum,
            root_objnum=1,
            size_val=size,
            offsets_type1=offsets_type1_base,
            compressed_map=compressed_base,
            prev=None,
            xref_obj_offset=base_xref_offset,
        )
        buf += b"startxref\n" + str(base_xref_offset).encode("ascii") + b"\n%%EOF\n"

        # Update revision: new object stream (7 0) and new xref stream (9 0)
        offsets_update: Dict[int, int] = {}
        if not buf.endswith(b"\n"):
            buf += b"\n"

        offsets_update[7] = len(buf)
        buf += build_objstm(7, objnums, rev=1)

        update_xref_objnum = 9
        update_xref_offset = len(buf)

        compressed_update: Dict[int, Tuple[int, int]] = {}
        for i, on in enumerate(objnums):
            compressed_update[on] = (7, i)

        offsets_type1_update = dict(offsets_base)
        offsets_type1_update[7] = offsets_update[7]
        offsets_type1_update[6] = offsets_base[6]
        offsets_type1_update[8] = base_xref_offset  # keep base xref object offset
        # object 9 offset will be set via xref_obj_offset

        buf += build_xref_stream_obj(
            xref_objnum=update_xref_objnum,
            root_objnum=1,
            size_val=size,
            offsets_type1=offsets_type1_update,
            compressed_map=compressed_update,
            prev=base_xref_offset,
            xref_obj_offset=update_xref_offset,
        )
        buf += b"startxref\n" + str(update_xref_offset).encode("ascii") + b"\n%%EOF\n"

        return bytes(buf)