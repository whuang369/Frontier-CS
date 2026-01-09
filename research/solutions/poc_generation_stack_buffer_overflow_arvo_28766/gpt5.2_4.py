import os
import re
import tarfile
import tempfile
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set


@dataclass
class FieldDef:
    name: str
    number: int
    label: str  # optional/repeated/required
    type_raw: str  # as written
    type_full: Optional[str] = None  # resolved full message name if message type else None
    is_enum: bool = False


class ProtoSchema:
    _re_block_comment = re.compile(r"/\*.*?\*/", re.DOTALL)
    _re_line_comment = re.compile(r"//[^\n]*")
    _re_token = re.compile(r'[A-Za-z_][A-Za-z0-9_\.]*|\d+|"(?:\\.|[^"\\])*"|[{}=;<>,\[\]]')

    def __init__(self):
        self.package_by_file: Dict[str, str] = {}
        self.messages: Dict[str, List[FieldDef]] = {}  # full_name -> fields
        self.enums: Set[str] = set()  # full_name for enums
        self.simple_to_full_msgs: Dict[str, List[str]] = {}
        self.simple_to_full_enums: Dict[str, List[str]] = {}
        self._defined_msgs: Set[str] = set()

    @staticmethod
    def _strip_comments(s: str) -> str:
        s = ProtoSchema._re_block_comment.sub("", s)
        s = ProtoSchema._re_line_comment.sub("", s)
        return s

    def _add_message(self, full_name: str):
        if full_name not in self.messages:
            self.messages[full_name] = []
        self._defined_msgs.add(full_name)
        simple = full_name.split(".")[-1]
        self.simple_to_full_msgs.setdefault(simple, []).append(full_name)

    def _add_enum(self, full_name: str):
        self.enums.add(full_name)
        simple = full_name.split(".")[-1]
        self.simple_to_full_enums.setdefault(simple, []).append(full_name)

    def parse_proto_file(self, path: str, text: str):
        text = self._strip_comments(text)
        tokens = self._re_token.findall(text)

        pkg = ""
        msg_stack: List[str] = []
        ctx_stack: List[str] = []  # message/enum/oneof
        i = 0

        def cur_msg_full() -> Optional[str]:
            if not msg_stack:
                return None
            if pkg:
                return pkg + "." + ".".join(msg_stack)
            return ".".join(msg_stack)

        while i < len(tokens):
            t = tokens[i]
            if t == "package":
                i += 1
                if i < len(tokens):
                    pkg = tokens[i].strip('"')
                while i < len(tokens) and tokens[i] != ";":
                    i += 1
                i += 1
                continue

            if t == "message":
                i += 1
                if i >= len(tokens):
                    break
                name = tokens[i]
                while i < len(tokens) and tokens[i] != "{":
                    i += 1
                if i >= len(tokens) or tokens[i] != "{":
                    i += 1
                    continue
                msg_stack.append(name)
                ctx_stack.append("message")
                fulln = cur_msg_full()
                if fulln:
                    self._add_message(fulln)
                i += 1
                continue

            if t == "enum":
                i += 1
                if i >= len(tokens):
                    break
                name = tokens[i]
                while i < len(tokens) and tokens[i] != "{":
                    i += 1
                if i >= len(tokens) or tokens[i] != "{":
                    i += 1
                    continue
                ctx_stack.append("enum")
                fulln = cur_msg_full()
                if fulln:
                    enum_full = fulln + "." + name
                else:
                    enum_full = (pkg + "." if pkg else "") + name
                self._add_enum(enum_full)
                i += 1
                continue

            if t == "oneof":
                while i < len(tokens) and tokens[i] != "{":
                    i += 1
                if i < len(tokens) and tokens[i] == "{":
                    ctx_stack.append("oneof")
                    i += 1
                else:
                    i += 1
                continue

            if t == "}":
                if ctx_stack:
                    kind = ctx_stack.pop()
                    if kind == "message":
                        if msg_stack:
                            msg_stack.pop()
                i += 1
                continue

            # Only parse fields while inside a message and not inside an enum.
            in_enum = "enum" in ctx_stack
            if not msg_stack or in_enum:
                i += 1
                continue

            # Skip statements starting with these keywords.
            if t in ("import", "option", "reserved", "extensions", "extend", "service", "rpc", "syntax"):
                while i < len(tokens) and tokens[i] != ";":
                    if tokens[i] == "{":
                        # skip block
                        depth = 1
                        i += 1
                        while i < len(tokens) and depth:
                            if tokens[i] == "{":
                                depth += 1
                            elif tokens[i] == "}":
                                depth -= 1
                            i += 1
                        break
                    i += 1
                if i < len(tokens) and tokens[i] == ";":
                    i += 1
                continue

            # Try parse a field definition: [label] type name = number ...
            label = "optional"
            type_tok = t
            if t in ("repeated", "optional", "required"):
                label = t
                i += 1
                if i >= len(tokens):
                    break
                type_tok = tokens[i]

            # map<...> not supported for our purposes; skip to ';'
            if type_tok == "map":
                while i < len(tokens) and tokens[i] != ";":
                    i += 1
                if i < len(tokens) and tokens[i] == ";":
                    i += 1
                else:
                    i += 1
                continue

            # Need type, name, '=', number, ';'
            if i + 4 >= len(tokens):
                i += 1
                continue
            type_name = type_tok
            name_tok = tokens[i + 1]
            eq_tok = tokens[i + 2]
            num_tok = tokens[i + 3]

            if eq_tok != "=" or not num_tok.isdigit():
                i += 1
                continue

            # Advance i to after number, then skip options until ';'
            field_no = int(num_tok)
            while i < len(tokens) and tokens[i] != ";":
                i += 1
            if i < len(tokens) and tokens[i] == ";":
                i += 1

            fulln = cur_msg_full()
            if fulln:
                self.messages[fulln].append(FieldDef(name=name_tok, number=field_no, label=label, type_raw=type_name))

        self.package_by_file[path] = pkg

    def _resolve_type_full(self, type_raw: str, context_msg_full: str) -> Optional[str]:
        # Returns full message name if type refers to a message; else None.
        if not type_raw:
            return None
        tr = type_raw
        if tr.startswith("."):
            tr = tr[1:]
            if tr in self._defined_msgs:
                return tr
            # allow missing package qualification mismatch by last component
            simp = tr.split(".")[-1]
            lst = self.simple_to_full_msgs.get(simp)
            if lst:
                return lst[0]
            return None

        # Scalar or built-in types
        scalar = {
            "double", "float", "int32", "int64", "uint32", "uint64", "sint32", "sint64",
            "fixed32", "fixed64", "sfixed32", "sfixed64", "bool", "string", "bytes",
        }
        if tr in scalar:
            return None

        # Relative qualification
        if "." in tr:
            # May be package-qualified without leading dot
            if tr in self._defined_msgs:
                return tr
            simp = tr.split(".")[-1]
            lst = self.simple_to_full_msgs.get(simp)
            if lst:
                return lst[0]
            return None

        # Try nested resolution: context ancestors
        ctx_parts = context_msg_full.split(".")
        # Remove the current message name progressively
        for cut in range(len(ctx_parts), 0, -1):
            cand = ".".join(ctx_parts[:cut] + [tr])
            if cand in self._defined_msgs:
                return cand

        # Try package + type
        pkg = ".".join(ctx_parts[:-1])
        if pkg:
            cand = pkg + "." + tr
            if cand in self._defined_msgs:
                return cand

        # Unique simple match
        lst = self.simple_to_full_msgs.get(tr)
        if lst:
            if len(lst) == 1:
                return lst[0]
            # If multiple, pick one in same package prefix if possible
            for f in lst:
                if pkg and f.startswith(pkg + "."):
                    return f
            return lst[0]
        return None

    def resolve_all_types(self):
        # Resolve message types and mark enum fields
        enum_simple_set = set(self.simple_to_full_enums.keys())
        for msg_full, fields in self.messages.items():
            for f in fields:
                f.type_full = self._resolve_type_full(f.type_raw, msg_full)
                if f.type_full is None:
                    # If it isn't scalar and matches an enum simple name, treat as enum
                    if f.type_raw in enum_simple_set or f.type_raw in self.enums:
                        f.is_enum = True
                    else:
                        # Also treat unknown non-scalar as enum (common for imported enums)
                        scalar = {
                            "double", "float", "int32", "int64", "uint32", "uint64", "sint32", "sint64",
                            "fixed32", "fixed64", "sfixed32", "sfixed64", "bool", "string", "bytes",
                        }
                        if f.type_raw not in scalar and f.type_raw and f.type_raw[0].isalpha():
                            # uncertain: could be message from other file not parsed; but we parse all .proto we find
                            if f.type_raw in enum_simple_set:
                                f.is_enum = True

    def get_message_fields(self, msg_full: str) -> List[FieldDef]:
        return self.messages.get(msg_full, [])

    def find_message_by_simple(self, simple: str) -> Optional[str]:
        lst = self.simple_to_full_msgs.get(simple)
        if not lst:
            return None
        if len(lst) == 1:
            return lst[0]
        return lst[0]


def encode_varint(x: int) -> bytes:
    if x < 0:
        x &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def zigzag64(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def zigzag32(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def encode_key(field_no: int, wire_type: int) -> bytes:
    return encode_varint((field_no << 3) | wire_type)


def encode_field_scalar(field_no: int, field_type: str, value: int) -> bytes:
    t = field_type
    if t in ("int32", "int64", "uint32", "uint64", "bool") or t == "" or t is None:
        return encode_key(field_no, 0) + encode_varint(int(value))
    if t in ("sint32",):
        return encode_key(field_no, 0) + encode_varint(zigzag32(int(value)))
    if t in ("sint64",):
        return encode_key(field_no, 0) + encode_varint(zigzag64(int(value)))
    if t in ("fixed64", "sfixed64", "double"):
        return encode_key(field_no, 1) + struct.pack("<Q", int(value) & ((1 << 64) - 1))
    if t in ("fixed32", "sfixed32", "float"):
        return encode_key(field_no, 5) + struct.pack("<I", int(value) & 0xFFFFFFFF)
    # enums are varint
    return encode_key(field_no, 0) + encode_varint(int(value))


def encode_field_bytes(field_no: int, b: bytes) -> bytes:
    return encode_key(field_no, 2) + encode_varint(len(b)) + b


def encode_field_string(field_no: int, s: str) -> bytes:
    b = s.encode("utf-8", "ignore")
    return encode_field_bytes(field_no, b)


def encode_message(fields_bytes: List[Tuple[int, bytes]]) -> bytes:
    # fields_bytes items: (field_no, encoded_field_bytes)
    fields_bytes.sort(key=lambda x: x[0])
    out = bytearray()
    for _, fb in fields_bytes:
        out += fb
    return bytes(out)


def walk_text_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    paths = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dp, f))
    return paths


def read_text(path: str, max_bytes: int = 4_000_000) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
        return data.decode("utf-8", "ignore")
    except Exception:
        return ""


def extract_pbzero_names_near_node_id_map(root: str) -> Set[str]:
    names: Set[str] = set()
    cpp_files = walk_text_files(root, (".cc", ".cpp", ".cxx", ".h", ".hpp"))
    for p in cpp_files:
        txt = read_text(p, max_bytes=2_000_000)
        if "node_id_map" not in txt:
            continue
        for m in re.finditer(r"pbzero::([A-Za-z_]\w*)", txt):
            names.add(m.group(1))
        for m in re.finditer(r"protos::pbzero::([A-Za-z_]\w*)", txt):
            names.add(m.group(1))
        for m in re.finditer(r"::Decoder\s+([A-Za-z_]\w*)\s*\(", txt):
            # not precise, but keep
            names.add(m.group(1))
    return names


def detect_input_root_kind(root: str) -> str:
    # returns: "trace_stream" (TracePacket stream), "trace_packet", "unknown"
    fuzz_files = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            fl = f.lower()
            if "fuzz" in fl or "fuzzer" in fl:
                if fl.endswith((".cc", ".cpp", ".cxx")):
                    fuzz_files.append(os.path.join(dp, f))
    # Also scan all sources for LLVMFuzzerTestOneInput
    for p in walk_text_files(root, (".cc", ".cpp", ".cxx")):
        txt = read_text(p, max_bytes=500_000)
        if "LLVMFuzzerTestOneInput" in txt:
            fuzz_files.append(p)

    seen = set()
    for p in fuzz_files:
        if p in seen:
            continue
        seen.add(p)
        txt = read_text(p, max_bytes=1_000_000)
        if "TraceProcessor" in txt or "trace_processor" in txt or "ParseTrace" in txt:
            return "trace_stream"
        if "TracePacket::Decoder" in txt or "pbzero::TracePacket" in txt:
            return "trace_packet"
        if "protos::TracePacket" in txt:
            return "trace_packet"
    return "trace_stream"


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(td)
            except Exception:
                # If not a tarball, assume it's a directory
                if os.path.isdir(src_path):
                    td = src_path
                else:
                    return b"\x0a\x00"

            root = td
            # If tarball extracted into a single top-level dir, use it
            try:
                entries = [os.path.join(td, x) for x in os.listdir(td)]
                dirs = [x for x in entries if os.path.isdir(x)]
                if len(dirs) == 1 and not any(os.path.isfile(x) for x in entries):
                    root = dirs[0]
            except Exception:
                pass

            schema = ProtoSchema()
            proto_files = walk_text_files(root, (".proto",))
            for p in proto_files:
                txt = read_text(p, max_bytes=6_000_000)
                if txt:
                    schema.parse_proto_file(p, txt)
            schema.resolve_all_types()

            pbzero_names = extract_pbzero_names_near_node_id_map(root)

            # Identify likely graph message with nodes and edges
            candidates: List[Tuple[int, str, FieldDef, FieldDef]] = []
            for msg_full, fields in schema.messages.items():
                rep_msgs = [f for f in fields if f.label == "repeated" and f.type_full is not None]
                if not rep_msgs:
                    continue
                node_field = None
                edge_field = None
                for f in rep_msgs:
                    simp = f.type_full.split(".")[-1]
                    nm = f.name.lower()
                    if node_field is None and ("node" in nm or simp.lower().endswith("node") or "node" in simp.lower()):
                        node_field = f
                    if edge_field is None and ("edge" in nm or simp.lower().endswith("edge") or "edge" in simp.lower()):
                        edge_field = f
                if node_field and edge_field:
                    score = 0
                    simp_msg = msg_full.split(".")[-1].lower()
                    if "memory" in simp_msg:
                        score += 20
                    if "snapshot" in simp_msg:
                        score += 20
                    if "graph" in simp_msg:
                        score += 10
                    if "heap" in simp_msg:
                        score += 5
                    if msg_full.split(".")[-1] in pbzero_names:
                        score += 80
                    if node_field.type_full.split(".")[-1] in pbzero_names:
                        score += 40
                    if edge_field.type_full.split(".")[-1] in pbzero_names:
                        score += 40
                    candidates.append((score, msg_full, node_field, edge_field))

            if not candidates:
                # fallback: try any message named MemoryGraph/HeapGraph
                for name in ("MemoryGraph", "HeapGraph", "MemorySnapshot", "HeapSnapshot"):
                    m = schema.find_message_by_simple(name)
                    if m:
                        candidates.append((1, m, FieldDef("nodes", 1, "repeated", "", None), FieldDef("edges", 2, "repeated", "", None)))
                        break
            if not candidates:
                return b"\x0a\x00"

            candidates.sort(key=lambda x: (-x[0], x[1]))
            graph_msg_full, nodes_field, edges_field = candidates[0][1], candidates[0][2], candidates[0][3]

            node_msg_full = nodes_field.type_full
            edge_msg_full = edges_field.type_full
            if not node_msg_full or not edge_msg_full:
                return b"\x0a\x00"

            node_fields = schema.get_message_fields(node_msg_full)
            edge_fields = schema.get_message_fields(edge_msg_full)

            def pick_id_field(fields: List[FieldDef], preferred_names: Tuple[str, ...]) -> Optional[FieldDef]:
                varint_like = {"int32", "int64", "uint32", "uint64", "sint32", "sint64", "bool"}
                best = None
                for pn in preferred_names:
                    for f in fields:
                        if f.name == pn or f.name.lower() == pn.lower():
                            if f.type_full is None and (f.type_raw in varint_like or f.is_enum or f.type_raw not in ("string", "bytes", "fixed32", "fixed64", "sfixed32", "sfixed64", "float", "double")):
                                return f
                # fallback: first varint-like numeric field
                for f in fields:
                    if f.type_full is None and f.type_raw in varint_like:
                        best = f
                        break
                if best:
                    return best
                # fallback: any scalar (including enum/unknown)
                for f in fields:
                    if f.type_full is None and f.type_raw not in ("string", "bytes"):
                        return f
                return None

            node_id_f = pick_id_field(node_fields, ("id", "node_id"))
            src_f = pick_id_field(edge_fields, ("source_node_id", "src_node_id", "source_id", "from_node_id", "from_id"))
            dst_f = pick_id_field(edge_fields, ("target_node_id", "dst_node_id", "target_id", "to_node_id", "to_id"))

            if node_id_f is None or src_f is None or dst_f is None:
                return b"\x0a\x00"

            # Build Node(id=1)
            node_msg = encode_message([
                (node_id_f.number, encode_field_scalar(node_id_f.number, node_id_f.type_raw, 1)),
            ])

            # Build Edge(source=1, target=2 missing)
            edge_msg = encode_message([
                (src_f.number, encode_field_scalar(src_f.number, src_f.type_raw, 1)),
                (dst_f.number, encode_field_scalar(dst_f.number, dst_f.type_raw, 2)),
            ])

            graph_msg = encode_message([
                (nodes_field.number, encode_field_bytes(nodes_field.number, node_msg)),
                (edges_field.number, encode_field_bytes(edges_field.number, edge_msg)),
            ])

            # Build message graph for BFS
            adj: Dict[str, List[Tuple[str, FieldDef]]] = {}
            for m, flds in schema.messages.items():
                for f in flds:
                    if f.type_full is not None:
                        adj.setdefault(m, []).append((f.type_full, f))

            def find_full_by_simple_pref(simple: str) -> Optional[str]:
                lst = schema.simple_to_full_msgs.get(simple)
                if not lst:
                    return None
                # prefer any in perfetto/protos-ish package
                for f in lst:
                    if "perfetto" in f.lower() or "protos" in f.lower():
                        return f
                return lst[0]

            tracepacket_full = find_full_by_simple_pref("TracePacket") or schema.find_message_by_simple("TracePacket")
            trace_full = find_full_by_simple_pref("Trace") or schema.find_message_by_simple("Trace")

            root_kind = detect_input_root_kind(root)

            # BFS from TracePacket to graph_msg_full to find wrapping chain
            def bfs_path(start: str, goal: str, max_depth: int = 8) -> Optional[List[Tuple[str, FieldDef, str]]]:
                # returns list of (parent_msg, field_def_used, child_msg) from start to goal
                if start == goal:
                    return []
                from collections import deque
                q = deque()
                q.append(start)
                prev: Dict[str, Tuple[str, FieldDef]] = {}
                depth: Dict[str, int] = {start: 0}
                while q:
                    cur = q.popleft()
                    d = depth[cur]
                    if d >= max_depth:
                        continue
                    for nxt, fld in adj.get(cur, []):
                        if nxt not in depth:
                            depth[nxt] = d + 1
                            prev[nxt] = (cur, fld)
                            if nxt == goal:
                                # reconstruct
                                path_rev = []
                                x = goal
                                while x != start:
                                    p, fd = prev[x]
                                    path_rev.append((p, fd, x))
                                    x = p
                                path_rev.reverse()
                                return path_rev
                            q.append(nxt)
                return None

            packet_bytes = b""
            if tracepacket_full:
                path = bfs_path(tracepacket_full, graph_msg_full)
                if path is None:
                    # Try alternative: sometimes graph is nested inside some other wrapper accessible from TracePacket
                    # If not reachable, just place graph directly in TracePacket as an unknown field using heuristic:
                    # Find any field in TracePacket whose type matches graph message
                    tp_fields = schema.get_message_fields(tracepacket_full)
                    chosen = None
                    for f in tp_fields:
                        if f.type_full == graph_msg_full:
                            chosen = f
                            break
                    if chosen is None:
                        # If can't, just return graph bytes (best-effort)
                        if root_kind == "trace_packet":
                            return graph_msg
                        # wrap as a trace packet stream with an empty packet (won't crash) fallback
                        return encode_key(1, 2) + encode_varint(len(graph_msg)) + graph_msg

                    tp_extra = []
                    for f in tp_fields:
                        if f.name.lower() == "timestamp":
                            tp_extra.append((f.number, encode_field_scalar(f.number, f.type_raw, 1)))
                        elif f.name.lower() == "trusted_packet_sequence_id":
                            tp_extra.append((f.number, encode_field_scalar(f.number, f.type_raw, 1)))
                    tp_fields_bytes = [(chosen.number, encode_field_bytes(chosen.number, graph_msg))]
                    for num, fb in tp_extra:
                        tp_fields_bytes.append((num, fb))
                    packet_bytes = encode_message(tp_fields_bytes)
                else:
                    inner = graph_msg
                    # wrap from bottom up
                    for parent, fld, child in reversed(path):
                        inner = encode_message([(fld.number, encode_field_bytes(fld.number, inner))])
                    # add optional TracePacket metadata if we're at TracePacket
                    tp_fields = schema.get_message_fields(tracepacket_full)
                    tp_extra = []
                    for f in tp_fields:
                        if f.name.lower() == "timestamp":
                            tp_extra.append((f.number, encode_field_scalar(f.number, f.type_raw, 1)))
                        elif f.name.lower() == "trusted_packet_sequence_id":
                            tp_extra.append((f.number, encode_field_scalar(f.number, f.type_raw, 1)))
                    if tp_extra:
                        fb = [(9999999, inner)]
                        # Merge: inner already is TracePacket bytes if path started at TracePacket,
                        # but our wrapping built full TracePacket only if start == TracePacket.
                        # Here, 'inner' is bytes for the field chain embedded inside TracePacket only if start != TracePacket.
                        # Since start is TracePacket, inner is full TracePacket content with only that chain set.
                        # So just append extras by concatenation (protobuf allows repeated fields and any order).
                        extras_b = encode_message([(n, b) for n, b in tp_extra])
                        packet_bytes = inner + extras_b
                    else:
                        packet_bytes = inner
            else:
                # No TracePacket in schema: output graph directly
                if root_kind == "trace_packet":
                    return graph_msg
                return encode_key(1, 2) + encode_varint(len(graph_msg)) + graph_msg

            if root_kind == "trace_packet":
                return packet_bytes

            # Build trace stream: repeated field in Trace which contains TracePacket
            packet_field_no = 1
            if trace_full and tracepacket_full:
                for f in schema.get_message_fields(trace_full):
                    if f.type_full == tracepacket_full and f.label == "repeated":
                        packet_field_no = f.number
                        break

            trace_stream = encode_key(packet_field_no, 2) + encode_varint(len(packet_bytes)) + packet_bytes
            return trace_stream