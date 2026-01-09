from typing import *

class Solution:
    def solve(self, src_path: str) -> bytes:
        lines = []
        lines.append('FILE "dummy.wav" WAVE')
        for track in range(1, 100):  # Tracks 01..99
            lines.append(f'  TRACK {track:02d} AUDIO')
            mm = track - 1
            lines.append(f'    INDEX 01 {mm:02d}:00:00')
        cuesheet = "\n".join(lines) + "\n"
        return cuesheet.encode("ascii")