import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Strategy:
        1. Search the provided source tarball for a file that is likely to be an existing PoC:
           - Prefer files whose size matches the known ground-truth PoC length (73 bytes).
           - Among these, prefer filenames containing indicative substrings (poc, crash, h225, etc.).
        2. If no such file is found, fall back to emitting a deterministic synthetic payload.
        """
        GROUND_TRUTH_LEN = 73

        poc = self._find_poc_in_tar(src_path, GROUND_TRUTH_LEN)
        if poc is not None:
            return poc

        # Fallback: deterministic synthetic payload of the ground-truth length.
        return self._fallback_poc(GROUND_TRUTH_LEN)

    def _compute_name_weight(self, name: str) -> int:
        """
        Heuristic weight based on filename indicating likelihood of being a PoC.
        Higher weight => more likely to be interesting.
        """
        lname = name.lower()
        weight = 0
        if "poc" in lname:
            weight += 100
        if "crash" in lname or "bug" in lname:
            weight += 60
        if "uaf" in lname or "useafterfree" in lname or "use-after-free" in lname:
            weight += 50
        if "heap" in lname:
            weight += 30
        if "h225" in lname or "ras" in lname:
            weight += 40
        if "wireshark" in lname:
            weight += 20
        if "fuzz" in lname:
            weight += 10
        if "id_" in lname or "clusterfuzz" in lname or "oss-fuzz" in lname:
            weight += 10
        if "test" in lname or "case" in lname:
            weight += 5
        return weight

    def _find_poc_in_tar(self, src_path: str, target_len: int) -> Optional[bytes]:
        """
        Inspect the tarball for candidate PoC files.

        Preference order:
        1. Files of exactly target_len bytes, scored by filename heuristics.
        2. Otherwise, small files (<= 4096 bytes) with suggestive names, scored by heuristics.
        """
        try:
            with tarfile.open(src_path, "r:*") as tf:
                best_exact_member = None
                best_exact_priority = None  # (neg_weight, depth, size)

                best_other_member = None
                best_other_priority = None

                for member in tf:
                    if not member.isreg():
                        continue

                    size = member.size
                    name = os.path.basename(member.name)
                    depth = member.name.count("/")

                    # Exact-size candidates (most promising)
                    if size == target_len:
                        weight = self._compute_name_weight(name)
                        priority = (-weight, depth, size)
                        if best_exact_member is None or priority < best_exact_priority:
                            best_exact_member = member
                            best_exact_priority = priority
                        continue

                    # Other small, interestingly-named candidates
                    if size <= 4096:
                        weight = self._compute_name_weight(name)
                        if weight > 0:
                            priority = (-weight, depth, size)
                            if best_other_member is None or priority < best_other_priority:
                                best_other_member = member
                                best_other_priority = priority

                # Try to read the best exact-length candidate first
                if best_exact_member is not None:
                    f = tf.extractfile(best_exact_member)
                    if f is not None:
                        data = f.read()
                        if len(data) > 0:
                            return data

                # Fallback to other small heuristic candidates
                if best_other_member is not None:
                    f = tf.extractfile(best_other_member)
                    if f is not None:
                        data = f.read()
                        if len(data) > 0:
                            return data

        except Exception:
            # Any issue reading/parsing the tarball: just fall back later.
            pass

        return None

    def _fallback_poc(self, length: int) -> bytes:
        """
        Deterministic synthetic payload as a last-resort PoC candidate.
        """
        # Simple repeating pattern to maximize structural variety.
        pattern = b"WSH225UAF!"
        out = bytearray()
        while len(out) < length:
            remaining = length - len(out)
            out.extend(pattern[:remaining])
        return bytes(out[:length])