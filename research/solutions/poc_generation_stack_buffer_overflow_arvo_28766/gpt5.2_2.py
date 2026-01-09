import os
import re
import json
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="pocgen_") as td:
            self._safe_extract_tar(src_path, td)

            sample_path = self._find_candidate_sample(td)
            if sample_path is not None:
                b = self._try_modify_sample(sample_path)
                if b is not None:
                    return b

            schema = self._infer_schema(td)
            if schema.get("format") == "json":
                return self._make_json_poc(schema)

            if schema.get("format") == "line":
                return self._make_line_poc(schema)

            return self._make_binary_poc(schema)

    def _safe_extract_tar(self, tar_path: str, dst_dir: str) -> Nonebytes:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not m.name or m.islnk() or m.issym():
                    continue
                name = m.name
                # Normalize and block path traversal
                if name.startswith("/") or name.startswith("\\"):
                    continue
                norm = os.path.normpath(name)
                if norm.startswith("..") or norm.startswith("../") or norm.startswith("..\\"):
                    continue
                out_path = os.path.join(dst_dir, norm)
                out_dir = os.path.dirname(out_path)
                os.makedirs(out_dir, exist_ok=True)
                try:
                    tf.extract(m, path=dst_dir, set_attrs=False)
                except Exception:
                    # Fallback: manual copy for safe cases
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        with open(out_path, "wb") as w:
                            w.write(f.read())
                    except Exception:
                        continue

    def _find_candidate_sample(self, root: str) -> Optional[str]:
        exts = {".json", ".snap", ".snapshot", ".txt", ".dat", ".bin", ".input"}
        best: Tuple[int, int, Optional[str]] = (-1, 10**18, None)  # (score, size, path)
        for dirpath, _, filenames in os.walk(root):
            dn = dirpath.lower()
            if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out")):
                continue
            for fn in filenames:
                low = fn.lower()
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except Exception:
                    continue
                if st.st_size <= 0 or st.st_size > 300_000:
                    continue
                ext = os.path.splitext(low)[1]
                if ext not in exts and not any(k in low for k in ("snapshot", "mem", "heap", "graph", "node", "edge", "corpus", "seed")):
                    continue

                score = 0
                name_hits = 0
                for k in ("snapshot", "mem", "heap", "graph", "node", "edge", "corpus", "seed"):
                    if k in low:
                        name_hits += 1
                score += 10 * name_hits

                if ext == ".json":
                    score += 15

                # Content sniff
                try:
                    with open(path, "rb") as f:
                        data = f.read(min(8192, st.st_size))
                except Exception:
                    continue

                dl = data.lower()
                for kw, s in (
                    (b"nodes", 30),
                    (b"edges", 30),
                    (b"node", 8),
                    (b"edge", 8),
                    (b"snapshot", 12),
                    (b"memory", 10),
                    (b"refs", 8),
                    (b"references", 8),
                    (b"node_id_map", 20),
                ):
                    if kw in dl:
                        score += s

                if score > best[0] or (score == best[0] and st.st_size < best[1]):
                    best = (score, st.st_size, path)

        return best[2]

    def _try_modify_sample(self, sample_path: str) -> Optional[bytes]:
        try:
            with open(sample_path, "rb") as f:
                data = f.read()
        except Exception:
            return None

        # Try JSON first
        try:
            txt = data.decode("utf-8")
        except Exception:
            txt = None

        if txt is not None:
            txt_stripped = txt.lstrip()
            if txt_stripped.startswith("{") or txt_stripped.startswith("["):
                try:
                    obj = json.loads(txt)
                    mod = self._modify_json_inplace(obj)
                    if mod:
                        out = json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
                        if not out.endswith("\n"):
                            out += "\n"
                        return out.encode("utf-8")
                except Exception:
                    pass

        # Try simple line/text modifications
        try:
            s = data.decode("latin-1")
        except Exception:
            return None

        if any(k in s.lower() for k in ("node", "edge", "snapshot", "graph", "refs", "reference")):
            # Heuristic: find two integers on a line that looks like an edge and change the second to a large missing id
            lines = s.splitlines(True)
            node_ids: Set[int] = set()
            for ln in lines:
                if re.search(r"\bnode\b", ln, flags=re.IGNORECASE):
                    m = re.search(r"\b(\d{1,10})\b", ln)
                    if m:
                        try:
                            node_ids.add(int(m.group(1)))
                        except Exception:
                            pass
            missing = self._pick_missing_id(node_ids)
            for i, ln in enumerate(lines):
                if re.search(r"\bedge\b", ln, flags=re.IGNORECASE) or re.search(r"\bref\b", ln, flags=re.IGNORECASE):
                    nums = list(re.finditer(r"\b\d{1,10}\b", ln))
                    if len(nums) >= 2:
                        a, b = nums[0], nums[1]
                        new_ln = ln[:b.start()] + str(missing) + ln[b.end():]
                        lines[i] = new_ln
                        out = "".join(lines)
                        return out.encode("latin-1")
            # If no edge-like line, try replace first occurrence of a reference-like key with number
            m = re.search(r"(?i)\b(to|to_id|target|dst|child|ref|reference)\b[^0-9]{0,20}(\d{1,10})", s)
            if m:
                start, end = m.span(2)
                out = s[:start] + str(self._pick_missing_id(set())) + s[end:]
                return out.encode("latin-1")

        return None

    def _infer_schema(self, root: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "format": None,
            "json_lib": None,
            "type_string": True,
            "name_string": True,
            "edge_type_string": True,
            "magic": None,
            "line_keywords": None,
        }

        # Find likely relevant source files
        src_files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            dn = dirpath.lower()
            if any(x in dn for x in ("/.git", "\\.git", "/build", "\\build", "/out", "\\out")):
                continue
            for fn in filenames:
                low = fn.lower()
                if os.path.splitext(low)[1] in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".inl", ".inc", ".py", ".rs", ".java"):
                    src_files.append(os.path.join(dirpath, fn))

        # Prefer files containing "node_id_map" or fuzzer entry
        prioritized: List[str] = []
        others: List[str] = []
        for p in src_files:
            lp = p.lower()
            if any(k in lp for k in ("fuzz", "fuzzer", "snapshot", "mem", "heap", "processor", "parse")):
                prioritized.append(p)
            else:
                others.append(p)

        def read_text(path: str, limit: int = 300_000) -> str:
            try:
                with open(path, "rb") as f:
                    b = f.read(limit)
                return b.decode("utf-8", errors="ignore")
            except Exception:
                return ""

        corpus: List[str] = []
        for p in prioritized + others:
            t = read_text(p)
            if not t:
                continue
            if "LLVMFuzzerTestOneInput" in t or "FuzzerTestOneInput" in t or "node_id_map" in t:
                corpus.append(t)
            if len(corpus) >= 30:
                break

        combined = "\n".join(corpus) if corpus else "\n".join(read_text(p) for p in (prioritized[:10] + others[:10]))

        low = combined.lower()

        if "nlohmann" in low or "json.hpp" in low or "json::parse" in low:
            info["format"] = "json"
            info["json_lib"] = "nlohmann"
        if "rapidjson" in low:
            info["format"] = "json"
            info["json_lib"] = "rapidjson"

        if info["format"] == "json":
            # Attempt to detect whether "type" is string
            info["type_string"] = self._detect_key_string_type(combined, "type", default=True)
            info["edge_type_string"] = self._detect_key_string_type(combined, "edge_type", default=info["type_string"])
            info["name_string"] = self._detect_key_string_type(combined, "name", default=True)

        # Check for line-based parsing
        if info["format"] is None:
            if any(k in low for k in ("std::getline", "getline(", "istringstream", "strtok", "scanf", "fscanf")) and any(
                k in low for k in ("node", "edge", "snapshot", "graph")
            ):
                info["format"] = "line"
                # extract potential keywords
                kws = set()
                for kw in ("NODE", "NODES", "EDGE", "EDGES", "REF", "REFS", "REFERENCE", "REFERENCES"):
                    if kw.lower() in low:
                        kws.add(kw)
                info["line_keywords"] = sorted(kws) if kws else None

        # Try to find magic
        if info["format"] is None:
            m = re.search(r'(?i)\b(kmagic|magic)\b[^"\n]{0,80}"([^"\n]{2,16})"', combined)
            if m:
                info["magic"] = m.group(2).encode("latin-1", errors="ignore")

        if info["format"] is None:
            # default to json guess; better chance
            info["format"] = "json"
            info["json_lib"] = "guess"
            info["type_string"] = True
            info["edge_type_string"] = True
            info["name_string"] = True

        return info

    def _detect_key_string_type(self, text: str, key: str, default: bool = True) -> bool:
        # Heuristic: If key is accessed with GetString or std::string, treat as string.
        # If accessed with GetInt/GetUint/get<int>, treat as int.
        # Search locally around occurrences of "key".
        pat = re.compile(r'["\']' + re.escape(key) + r'["\']')
        hits = list(pat.finditer(text))
        if not hits:
            return default
        for h in hits[:20]:
            w = text[max(0, h.start() - 120): h.end() + 200]
            wl = w.lower()
            if "getstring" in wl or "std::string" in wl or "get<std::string" in wl:
                return True
            if "getint" in wl or "getuint" in wl or "get<uint" in wl or "get<int" in wl or "asint" in wl or "asuint" in wl:
                return False
        return default

    def _collect_ints_by_keys(self, obj: Any, key_names: Set[str], out: Set[int]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in key_names:
                    if isinstance(v, bool):
                        pass
                    elif isinstance(v, int):
                        out.add(v)
                    elif isinstance(v, str) and v.isdigit():
                        try:
                            out.add(int(v))
                        except Exception:
                            pass
                self._collect_ints_by_keys(v, key_names, out)
        elif isinstance(obj, list):
            for it in obj:
                self._collect_ints_by_keys(it, key_names, out)

    def _pick_missing_id(self, node_ids: Set[int]) -> int:
        for cand in (0x1337, 0x4242, 0x7FFFFFFF, 0x12345678, 99999999, 2, 3, 4):
            if cand not in node_ids:
                return cand
        x = 1
        while x in node_ids:
            x += 1
        return x

    def _modify_json_inplace(self, obj: Any) -> bool:
        node_id_keys = {"id", "node_id", "nodeid", "nodeID".lower(), "nid", "node"}
        # We will collect node ids based on typical keys, but avoid catching edge ids by limiting to nodes subtree if possible.
        node_ids: Set[int] = set()

        nodes_sub = None
        if isinstance(obj, dict) and "nodes" in obj:
            nodes_sub = obj.get("nodes")
        elif isinstance(obj, dict):
            for k in ("Nodes", "node", "Node", "node_list", "nodeList", "graph_nodes", "graphNodes"):
                if k in obj:
                    nodes_sub = obj.get(k)
                    break

        if nodes_sub is not None:
            self._collect_ints_by_keys(nodes_sub, node_id_keys, node_ids)
        else:
            self._collect_ints_by_keys(obj, node_id_keys, node_ids)

        missing = self._pick_missing_id(node_ids)

        # Prefer modifying a reference-like field (to/target/dst/ref...)
        ref_key_indicators = (
            "to",
            "to_id",
            "toid",
            "target",
            "target_id",
            "targetid",
            "dst",
            "dst_id",
            "dstid",
            "child",
            "child_id",
            "childid",
            "ref",
            "ref_id",
            "reference",
            "reference_id",
            "points_to",
            "pointsTo",
        )

        def is_ref_key(k: str) -> bool:
            kl = k.lower()
            if kl in ("id", "node_id", "nodeid", "from", "from_id", "fromid"):
                return False
            for ind in ref_key_indicators:
                if ind in kl:
                    return True
            return False

        def modify_first_reference(x: Any) -> bool:
            if isinstance(x, dict):
                for k in list(x.keys()):
                    v = x[k]
                    if isinstance(k, str) and is_ref_key(k):
                        if isinstance(v, int) and not isinstance(v, bool):
                            x[k] = missing
                            return True
                        if isinstance(v, str) and v.isdigit():
                            x[k] = str(missing)
                            return True
                        if isinstance(v, list) and v and all(isinstance(e, int) for e in v):
                            x[k][0] = missing
                            return True
                    if modify_first_reference(v):
                        return True
            elif isinstance(x, list):
                for it in x:
                    if modify_first_reference(it):
                        return True
            return False

        if modify_first_reference(obj):
            return True

        # If no obvious reference fields, inject a new edge/ref in a plausible place
        if isinstance(obj, dict):
            # Inject edge
            for edges_key in ("edges", "Edges", "references", "refs", "Refs", "links", "Links"):
                if edges_key in obj and isinstance(obj[edges_key], list):
                    from_id = next(iter(node_ids), 1)
                    obj[edges_key].append({"from": from_id, "to": missing})
                    return True
            # Inject refs into first node
            if "nodes" in obj and isinstance(obj["nodes"], list) and obj["nodes"]:
                n0 = obj["nodes"][0]
                if isinstance(n0, dict):
                    if "refs" in n0 and isinstance(n0["refs"], list):
                        n0["refs"].append(missing)
                    else:
                        n0["refs"] = [missing]
                    return True
        return False

    def _make_json_poc(self, schema: Dict[str, Any]) -> bytes:
        type_string = bool(schema.get("type_string", True))
        edge_type_string = bool(schema.get("edge_type_string", True))
        name_string = bool(schema.get("name_string", True))

        def tval(is_str: bool, sval: str = "object") -> Any:
            return sval if is_str else 0

        def nval(is_str: bool, sval: str = "root") -> Any:
            return sval if is_str else 0

        # A broadly compatible structure:
        # - top-level nodes array with node id
        # - edges array referencing a missing node id
        # - also embed refs in node
        doc: Dict[str, Any] = {
            "version": 1,
            "snapshot": {"type": "memory", "meta": {}},
            "nodes": [
                {
                    "id": 1,
                    "node_id": 1,
                    "type": tval(type_string, "Node"),
                    "name": nval(name_string, "root"),
                    "size": 0,
                    "self_size": 0,
                    "edge_count": 1,
                    "refs": [2],
                    "references": [2],
                }
            ],
            "edges": [
                {
                    "from": 1,
                    "from_id": 1,
                    "to": 2,
                    "to_id": 2,
                    "type": tval(edge_type_string, "ref"),
                    "name": nval(name_string, "x"),
                }
            ],
            "strings": ["root", "x", "Node", "ref"],
        }

        out = json.dumps(doc, separators=(",", ":"), ensure_ascii=True) + "\n"
        return out.encode("utf-8")

    def _make_line_poc(self, schema: Dict[str, Any]) -> bytes:
        # Generic graph-like text with an invalid reference
        # Try to include common keywords if detected.
        kws = schema.get("line_keywords") or []
        use_nodes = "NODES" if "NODES" in kws else "nodes"
        use_edges = "EDGES" if "EDGES" in kws else "edges"
        use_node = "NODE" if "NODE" in kws else "node"
        use_edge = "EDGE" if "EDGE" in kws else "edge"
        use_refs = "REFS" if "REFS" in kws else ("refs" if "REF" not in kws else "ref")

        s = (
            f"{use_nodes} 1\n"
            f"{use_node} 1 root\n"
            f"{use_edges} 1\n"
            f"{use_edge} 1 2 x\n"
            f"{use_refs} 1 2\n"
        )
        return s.encode("utf-8")

    def _make_binary_poc(self, schema: Dict[str, Any]) -> bytes:
        # Fallback: simple little-endian graph:
        # [magic?][u32 node_count][u32 node_id...][u32 edge_count][u32 from][u32 to]
        magic = schema.get("magic")
        if not isinstance(magic, (bytes, bytearray)):
            magic = b""
        def u32(x: int) -> bytes:
            return int(x & 0xFFFFFFFF).to_bytes(4, "little", signed=False)

        b = bytearray()
        b += magic
        b += u32(1)      # node_count
        b += u32(1)      # node_id
        b += u32(1)      # edge_count
        b += u32(1)      # from_id
        b += u32(2)      # to_id (missing)
        return bytes(b)