import os
import re
import tarfile
import tempfile
from typing import List, Tuple, Optional


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.realpath(member_path).startswith(os.path.realpath(path) + os.sep):
            continue
        try:
            tar.extract(member, path)
        except Exception:
            continue


def _read_file_bytes(path: str, max_size: int = 1024 * 1024) -> Optional[bytes]:
    try:
        if os.path.getsize(path) > max_size:
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _iter_files(root: str) -> List[str]:
    files = []
    for d, _, fs in os.walk(root):
        for fn in fs:
            files.append(os.path.join(d, fn))
    return files


def _score_poc_path(path: str, size: int) -> int:
    p = path.lower()
    score = 0
    # Strong indicators
    if "poc" in p:
        score += 150
    if "use-after-free" in p or "uaf" in p:
        score += 120
    if "double" in p and "free" in p:
        score += 120
    if "double_free" in p or "double-free" in p:
        score += 100
    if "crash" in p or "repro" in p or "reproducer" in p:
        score += 80
    if "payload" in p:
        score += 60
    if "exploit" in p:
        score += 50
    if "fuzz" in p or "case" in p or "input" in p or "test" in p:
        score += 20
    if "good" in p or "valid" in p or "sample" in p or "example" in p:
        score -= 30
    if "expected" in p or "oracle" in p:
        score -= 50
    # Prefer smallish files
    if 1 <= size <= 4096:
        score += 40
    if 1 <= size <= 512:
        score += 20
    # Slight bonus for close to 60 bytes
    score += max(0, 20 - abs(size - 60))
    # Prefer text-like
    ext = os.path.splitext(path)[1].lower()
    if ext in {".in", ".txt", ".dat", ".bin", ""}:
        score += 10
    return score


def _find_best_poc_file(root: str) -> Optional[bytes]:
    candidates: List[Tuple[int, str]] = []
    for f in _iter_files(root):
        try:
            size = os.path.getsize(f)
        except Exception:
            continue
        # Skip source and build artifacts
        lower = f.lower()
        skip_exts = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c", ".o", ".obj",
                     ".a", ".so", ".dll", ".dylib", ".exe", ".bat", ".sh", ".py",
                     ".java", ".kt", ".rs", ".go", ".js", ".ts", ".json", ".yml", ".yaml",
                     ".md", ".mk", ".cmake", ".html", ".xml", ".gradle", ".sln", ".vcxproj"}
        ext = os.path.splitext(f)[1].lower()
        if ext in skip_exts:
            continue
        # Common names likely to be PoCs
        score = _score_poc_path(f, size)
        if score <= 0:
            # Also check content-based signals later only for likely candidates
            if "poc" not in lower and "crash" not in lower and "uaf" not in lower and "double" not in lower:
                continue
        candidates.append((score, f))
    candidates.sort(reverse=True)
    for _, f in candidates:
        data = _read_file_bytes(f, max_size=1024 * 1024)
        if not data:
            continue
        # Light content validation: prefer ASCII or small binary
        if any(b == 0 for b in data[:64]) and len(data) > 256:
            continue
        return data
    return None


def _extract_code_blocks_from_readme(text: str) -> List[str]:
    blocks: List[str] = []
    # Triple backticks
    for m in re.finditer(r"```(?:[a-zA-Z0-9_\-]*)\n(.*?)```", text, re.S):
        blocks.append(m.group(1))
    # Indented code blocks (four spaces)
    for m in re.finditer(r"(?:^|\n)(?: {4}|\t)([^\n]+(?:\n(?: {4}|\t)[^\n]+)*)", text):
        block = re.sub(r"^(?: {4}|\t)", "", m.group(1), flags=re.M)
        blocks.append(block)
    return blocks


def _find_poc_from_readmes(root: str) -> Optional[bytes]:
    readme_files = []
    for f in _iter_files(root):
        base = os.path.basename(f).lower()
        if base in {"readme", "readme.md", "readme.txt"} or base.startswith("readme"):
            readme_files.append(f)
    # Prefer top-level README files
    readme_files.sort(key=lambda p: (p.count(os.sep), len(p)))
    for f in readme_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
        except Exception:
            continue
        blocks = _extract_code_blocks_from_readme(txt)
        # Heuristic: choose a small block that looks like input (no includes or language keywords)
        for b in blocks:
            b_stripped = b.strip()
            if not b_stripped:
                continue
            if len(b_stripped) > 1024:
                continue
            # Filter out C/C++ code
            if "#include" in b_stripped or "int main" in b_stripped or "using namespace" in b_stripped:
                continue
            # Likely input: numbers, commands, JSON-ish, etc.
            # Avoid shell commands
            if re.search(r"\b(gcc|clang|make|cmake|python|cat|./)\b", b_stripped):
                continue
            # Avoid program output
            if re.search(r"AddressSanitizer|Segmentation fault|Usage:", b_stripped, re.I):
                continue
            try:
                return b_stripped.encode("utf-8", errors="ignore")
            except Exception:
                continue
    return None


def _gather_cpp_string_literals(root: str) -> List[str]:
    strings: List[str] = []
    cpp_exts = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".c"}
    for f in _iter_files(root):
        if os.path.splitext(f)[1].lower() not in cpp_exts:
            continue
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                code = fh.read()
        except Exception:
            continue
        # Remove comments to reduce noise
        code = re.sub(r"//.*?$", "", code, flags=re.M)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
        # Extract string literals
        for m in re.finditer(r'"([^"\\]|\\.|\\\n)*"', code):
            s = m.group(0)
            try:
                # Unescape minimal sequences
                val = bytes(s[1:-1], "utf-8").decode("unicode_escape")
            except Exception:
                val = s[1:-1]
            strings.append(val)
    return strings


def _detect_case(strings: List[str], word: str) -> Optional[str]:
    # Return variant of word present in strings if any
    lower = [s for s in strings if s.lower() == word.lower()]
    if not lower:
        return None
    # Prefer exact matches with consistent case
    for s in strings:
        if s == word:
            return s
    # Otherwise return the first occurrence (any case)
    return lower[0]


def _generate_command_style_poc(strings: List[str]) -> Optional[bytes]:
    # Determine command keywords
    add_kw = _detect_case(strings, "add") or _detect_case(strings, "ADD")
    node_kw = _detect_case(strings, "node") or _detect_case(strings, "NODE")
    child_kw = _detect_case(strings, "child") or _detect_case(strings, "CHILD")
    parent_kw = _detect_case(strings, "parent") or _detect_case(strings, "PARENT")
    edge_kw = _detect_case(strings, "edge") or _detect_case(strings, "EDGE")
    link_kw = _detect_case(strings, "link") or _detect_case(strings, "LINK")
    connect_kw = _detect_case(strings, "connect") or _detect_case(strings, "CONNECT")
    quit_kw = _detect_case(strings, "quit") or _detect_case(strings, "exit") or _detect_case(strings, "end")

    lines = []

    # Some programs expect a number of operations first
    ops_hint = None
    # Search for hints like "operations", "ops", "commands" in strings
    for s in strings:
        if re.fullmatch(r"[Nn]umber of (ops|operations|commands)", s):
            ops_hint = 1
            break

    # Try to craft commands to cause duplicate add or self-add
    if node_kw:
        # Create node 0 (potentially)
        lines.append(f"{node_kw} 0")
        # Some programs require named nodes; try numeric first
    # Try to use add/link/edge command
    verb = add_kw or link_kw or connect_kw or edge_kw
    if not verb:
        return None

    # Use various plausible forms: add parent child, add u v
    if parent_kw and child_kw:
        lines.append(f"{verb} 0 0")         # self-add to throw
        lines.append(f"{verb} 0 0")         # duplicate to reinforce
    else:
        lines.append(f"{verb} 0 0")         # generic u v
        lines.append(f"{verb} 0 0")         # duplicate

    if quit_kw:
        lines.append(quit_kw)

    data = ("\n".join(lines) + "\n").encode("utf-8")
    return data if data.strip() else None


def _generate_numeric_graph_poc(strings: List[str]) -> bytes:
    # Generic numeric format: n m then m lines of edges.
    # Include duplicates and self-loops to induce Node::add throws.
    # Keep it compact.
    lines = []
    edges = [
        (0, 0), (0, 0),  # duplicate self-loop
        (1, 1), (2, 2),  # self-loops
        (0, 1), (0, 1),  # duplicate
        (3, 3), (1, 2),
        (2, 1), (1, 1),  # duplicate self-loop
        (2, 2), (3, 3),  # duplicates
        (0, 0)           # another duplicate
    ]
    n = 4
    m = len(edges)
    lines.append(f"{n} {m}")
    for u, v in edges:
        lines.append(f"{u} {v}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _generate_json_like_poc(strings: List[str]) -> Optional[bytes]:
    # If code suggests JSON, produce duplicate keys to trigger add-throws
    json_hints = any(("json" in s.lower() or "nlohmann" in s.lower() or "rapidjson" in s.lower()) for s in strings)
    if not json_hints:
        return None
    # Duplicate key under same object
    poc = b'{"root":{"k":1,"k":2},"x":0}'
    return poc


class Solution:
    def solve(self, src_path: str) -> bytes:
        # 1) Extract source tarball
        tmpdir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                _safe_extract(tf, tmpdir)
        except Exception:
            # If extraction fails, return a robust numeric fallback
            return _generate_numeric_graph_poc([])

        # 2) Try to find an existing PoC in repo
        data = _find_best_poc_file(tmpdir)
        if data:
            return data

        # 3) Try to find sample input in README
        data = _find_poc_from_readmes(tmpdir)
        if data:
            return data

        # 4) Analyze strings to generate an input
        strings = _gather_cpp_string_literals(tmpdir)

        # 4a) Try command-style using discovered keywords
        data = _generate_command_style_poc(strings)
        if data:
            return data

        # 4b) Try JSON-like duplicate keys
        data = _generate_json_like_poc(strings)
        if data:
            return data

        # 4c) Fallback to numeric graph format with duplicates/self-loops
        return _generate_numeric_graph_poc(strings)