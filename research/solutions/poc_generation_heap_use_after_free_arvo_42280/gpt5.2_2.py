import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple, Iterable


class Solution:
    def _iter_source_texts(self, src_path: str) -> Iterable[Tuple[str, str]]:
        max_file_size = 2_000_000
        max_total = 80_000_000
        total = 0

        def should_read(name: str) -> bool:
            lname = name.lower()
            for ext in (".c", ".h", ".cc", ".cpp", ".inc", ".def", ".ps", ".txt"):
                if lname.endswith(ext):
                    return True
            if "/resource/" in lname or "/init/" in lname:
                if lname.endswith(".ps"):
                    return True
            return False

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    path = os.path.join(root, fn)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > max_file_size:
                        continue
                    rel = os.path.relpath(path, src_path).replace(os.sep, "/")
                    if not should_read(rel):
                        continue
                    if total + st.st_size > max_total:
                        return
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    total += len(data)
                    try:
                        text = data.decode("latin-1", errors="ignore")
                    except Exception:
                        continue
                    yield rel, text
            return

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return

        with tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.size <= 0 or m.size > max_file_size:
                    continue
                name = m.name
                if not should_read(name):
                    continue
                if total + m.size > max_total:
                    return
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                total += len(data)
                try:
                    text = data.decode("latin-1", errors="ignore")
                except Exception:
                    continue
                yield name, text

    def _safe_ps_name(self, name: str) -> bool:
        if not name:
            return False
        if name[0] == "%":
            return False
        if any(ch.isspace() for ch in name):
            return False
        if any(ch in name for ch in "()/{}[]<>"):
            return False
        if "/" in name:
            return False
        return True

    def _extract_ops(self, src_path: str) -> Dict[str, Optional[int]]:
        op_re = re.compile(r'\{\s*"([^"]+)"\s*,\s*([A-Za-z_][0-9A-Za-z_]*)\s*\}')
        ops: Dict[str, Optional[int]] = {}

        for _, text in self._iter_source_texts(src_path):
            for m in op_re.finditer(text):
                raw = m.group(1)
                if not raw:
                    continue
                argc = None
                name = raw
                mm = re.match(r"^(\d+)(.+)$", raw)
                if mm:
                    try:
                        argc = int(mm.group(1))
                    except Exception:
                        argc = None
                    name = mm.group(2)
                name = name.strip()
                if not self._safe_ps_name(name):
                    continue
                prev = ops.get(name, None)
                if prev is None:
                    ops[name] = argc
                else:
                    if argc is not None and (prev is None or argc < prev):
                        ops[name] = argc
        return ops

    def _pick(self, ops: Dict[str, Optional[int]]) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Optional[int]]]:
        names = list(ops.keys())
        lname_map = {n: n.lower() for n in names}

        begin: List[str] = []
        end: List[str] = []
        setter: List[str] = []
        use: List[str] = []

        def add_unique(lst: List[str], n: str):
            if n not in lst:
                lst.append(n)

        for n in names:
            ln = lname_map[n]
            if "runpdfbegin" in ln or (("pdfi" in ln) and (("begin" in ln) or ("start" in ln) or ("open" in ln))):
                add_unique(begin, n)
            if "runpdfend" in ln or (("pdfi" in ln) and ("end" in ln or "close" in ln or "finish" in ln)):
                add_unique(end, n)
            if "pdfi" in ln and (("stream" in ln) or ("input" in ln) or ("file" in ln)) and (("set" in ln) or ("input" in ln) or ("open" in ln)):
                add_unique(setter, n)

        for n in names:
            ln = lname_map[n]
            argc = ops.get(n, None)
            if argc == 0 and ("pdfi" in ln or "runpdf" in ln or (ln.startswith(".pdf") and "pdfi" in "".join(lname_map.values()))):
                if any(k in ln for k in ("begin", "start", "init", "set", "open", "close", "end", "finish", "flush", "free", "debug")):
                    continue
                add_unique(use, n)

        def sort_key(n: str) -> Tuple[int, int, str]:
            ln = lname_map[n]
            pri = 50
            if "runpdfbegin" in ln:
                pri = 0
            elif ln.startswith(".runpdfbegin"):
                pri = 1
            elif "pdfi" in ln and "begin" in ln:
                pri = 5
            return (pri, 0 if ln.startswith(".") else 1, n)

        begin.sort(key=sort_key)
        end.sort(key=sort_key)

        def sort_set(n: str) -> Tuple[int, int, str]:
            ln = lname_map[n]
            pri = 50
            if "set" in ln and "input" in ln and "stream" in ln:
                pri = 0
            elif "set" in ln and ("stream" in ln or "input" in ln):
                pri = 5
            return (pri, 0 if ln.startswith(".") else 1, n)

        setter.sort(key=sort_set)

        def sort_use(n: str) -> Tuple[int, int, str]:
            ln = lname_map[n]
            pri = 50
            if any(k in ln for k in ("token", "parse", "read", "exec", "run", "xref", "object", "obj")):
                pri = 0
            return (pri, 0 if ln.startswith(".") else 1, n)

        use.sort(key=sort_use)

        begin = begin[:6]
        setter = setter[:8]
        end = end[:6]
        use = use[:60]

        return begin, setter, use, end, ops

    def _gen_args(self, opname: str, argc: Optional[int]) -> str:
        if argc is None:
            return ""
        ln = opname.lower()
        if argc <= 0:
            return ""
        if argc == 1:
            if any(k in ln for k in ("stream", "file", "input")):
                return "F "
            if "dict" in ln:
                return "<< >> "
            if any(k in ln for k in ("name", "key")):
                return "/A "
            if any(k in ln for k in ("string", "path", "fname")):
                return "(A) "
            return "0 "
        if argc == 2:
            if any(k in ln for k in ("stream", "file", "input")):
                return "F 0 "
            if "dict" in ln:
                return "<< >> 0 "
            return "0 0 "
        if argc == 3:
            if any(k in ln for k in ("stream", "file", "input")):
                return "F 0 0 "
            return "0 0 0 "
        return ""

    def solve(self, src_path: str) -> bytes:
        ops = self._extract_ops(src_path)
        begin_ops, setter_ops, use_ops, end_ops, ops_argc = self._pick(ops)

        if not begin_ops and not setter_ops:
            begin_ops = [".runpdfbegin", "runpdfbegin", ".pdfi_begin", "pdfi_begin"]
            setter_ops = [".pdfi_set_input_stream", ".pdfi_setstream", ".setpdfinput"]
            use_ops = [".pdfi_exec", ".pdfi_token", ".pdfi_next_token", ".pdfi_parse"]
            end_ops = [".runpdfend", "runpdfend", ".pdfi_end", "pdfi_end"]

        lines: List[str] = []
        lines.append("%!PS-Adobe-3.0")
        lines.append("%%Title: arvo-42280")
        lines.append("/F null def")
        lines.append("{ (%stdout) (w) file /F exch def } stopped pop")
        lines.append("F null eq { { (%stderr) (w) file /F exch def } stopped pop } if")
        lines.append("F null eq { { (%stdin) (r) file /ASCII85Encode filter /F exch def } stopped pop } if")
        lines.append("/A 0 def")

        def emit_call(opname: str, argc: Optional[int], prefix_args: Optional[str] = None):
            if not opname or not self._safe_ps_name(opname):
                return
            args = self._gen_args(opname, argc)
            if prefix_args is not None:
                args = prefix_args
            lines.append(f"/{opname} where {{ pop {{ {args}{opname} }} stopped pop }} if")

        for b in begin_ops:
            emit_call(b, ops_argc.get(b, 1))

        for s in setter_ops:
            argc = ops_argc.get(s, 1)
            emit_call(s, argc, prefix_args="F ")

        for u in use_ops:
            argc = ops_argc.get(u, 0)
            if argc is None or argc > 3:
                continue
            emit_call(u, argc)

        for e in end_ops:
            emit_call(e, ops_argc.get(e, 0))

        lines.append("/pdfdict where { pop pdfdict begin")
        lines.append("  /pdfpagecount where { pop { pdfpagecount } stopped pop } if")
        lines.append("  /pdfgetpage where { pop { 1 pdfgetpage } stopped pop } if")
        lines.append("  /pdfshowpage where { pop { pdfshowpage } stopped pop } if")
        lines.append("end } if")

        lines.append("quit")
        data = "\n".join(lines).encode("latin-1", errors="ignore") + b"\n"
        return data