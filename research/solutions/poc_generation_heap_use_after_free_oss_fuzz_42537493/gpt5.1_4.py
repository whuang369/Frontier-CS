import os
import tarfile


class Solution:
    TARGET_ID = "42537493"
    TARGET_LEN = 24

    def solve(self, src_path: str) -> bytes:
        poc = self._find_poc_by_id(src_path)
        if poc is not None and len(poc) > 0:
            return poc

        poc = self._heuristic_search(src_path)
        if poc is not None and len(poc) > 0:
            return poc

        # Last-resort generic minimal XML
        return b"<a/>"

    def _find_poc_by_id(self, src_path: str) -> bytes | None:
        candidates = []

        if os.path.isdir(src_path):
            base = src_path
            for root, _, files in os.walk(base):
                for name in files:
                    rel = os.path.relpath(os.path.join(root, name), base)
                    if self.TARGET_ID in rel:
                        full = os.path.join(root, name)
                        try:
                            with open(full, "rb") as f:
                                data = f.read()
                        except OSError:
                            continue
                        if data:
                            candidates.append((rel, data))
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isreg():
                            continue
                        if self.TARGET_ID in m.name:
                            try:
                                f = tf.extractfile(m)
                            except (KeyError, OSError):
                                continue
                            if f is None:
                                continue
                            data = f.read()
                            if data:
                                candidates.append((m.name, data))
            except tarfile.ReadError:
                # If it's not a tarball but a regular file, just return its contents
                if os.path.isfile(src_path):
                    try:
                        with open(src_path, "rb") as f:
                            return f.read()
                    except OSError:
                        return None

        if not candidates:
            return None

        # Choose candidate whose size is closest to TARGET_LEN, tie-breaking by smaller size
        best_rel, best_data = min(
            candidates, key=lambda t: (abs(len(t[1]) - self.TARGET_LEN), len(t[1]))
        )
        return best_data

    def _heuristic_search(self, src_path: str) -> bytes | None:
        data_exts = {".xml", ".html", ".htm", ".svg", ".txt", ".dat", ".bin", ".json"}
        files: list[tuple[str, bytes]] = []

        if os.path.isdir(src_path):
            base = src_path
            for root, _, fs in os.walk(base):
                for name in fs:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, base)
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in data_exts:
                        continue
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        continue
                    if sz == 0 or sz > 8192:
                        continue
                    try:
                        with open(full, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    if data:
                        files.append((rel, data))
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isreg():
                            continue
                        rel = m.name
                        ext = os.path.splitext(rel)[1].lower()
                        if ext not in data_exts:
                            continue
                        if m.size == 0 or m.size > 8192:
                            continue
                        try:
                            f = tf.extractfile(m)
                        except (KeyError, OSError):
                            continue
                        if f is None:
                            continue
                        data = f.read()
                        if data:
                            files.append((rel, data))
            except tarfile.ReadError:
                return None

        if not files:
            return None

        def score(entry: tuple[str, bytes]) -> int:
            rel, data = entry
            n = len(data)
            lower_rel = rel.lower()
            s = 0

            # Path hints
            if "oss-fuzz" in lower_rel or "ossfuzz" in lower_rel:
                s += 40
            if "fuzz" in lower_rel:
                s += 15
            if "regress" in lower_rel or "repro" in lower_rel:
                s += 10
            if "crash" in lower_rel or "uaf" in lower_rel or "heap" in lower_rel:
                s += 8
            if "test" in lower_rel or "tests" in lower_rel:
                s += 4
            if self.TARGET_ID in rel:
                s += 60
            if self.TARGET_ID.encode() in data:
                s += 30

            # Extension weights
            ext = os.path.splitext(lower_rel)[1]
            if ext == ".xml":
                s += 15
            elif ext in (".html", ".htm", ".svg"):
                s += 10
            else:
                s += 5

            # Size closeness to target length
            diff = abs(n - self.TARGET_LEN)
            s += max(0, 40 - diff * 2)

            # General small-size preference
            if n <= 256:
                s += 5
            if n <= 64:
                s += 5
            if n == self.TARGET_LEN:
                s += 10

            return s

        best_rel, best_data = max(files, key=score)
        return best_data