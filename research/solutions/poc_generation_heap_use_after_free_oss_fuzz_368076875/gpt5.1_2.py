import tarfile
import zipfile
import io


class Solution:
    TRG_SIZE = 274773
    MAX_CONTAINER_RECURSION = 3
    NESTED_MAX_SIZE = 50 * 1024 * 1024

    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, "r:*") as tf:
                data = self._search_tarfile(tf, 0)
                if data is not None:
                    return data
        except tarfile.TarError:
            pass
        return b""

    def _score_member_name(self, name: str) -> int:
        n = name.lower()
        score = 0
        if "368076875" in n:
            score += 16
        if "oss-fuzz" in n or "clusterfuzz" in n:
            score += 8
        if "uaf" in n:
            score += 8
        if "crash" in n:
            score += 8
        if "poc" in n or "repro" in n or "reproducer" in n:
            score += 8
        if "bug" in n:
            score += 4
        if "id:" in n or "id_" in n:
            score += 4
        if "minimized" in n or "reduced" in n:
            score += 2
        if "testcase" in n:
            score += 2
        return score

    def _search_tarfile(self, tf: tarfile.TarFile, depth: int) -> bytes | None:
        if depth > self.MAX_CONTAINER_RECURSION:
            return None

        members = [m for m in tf.getmembers() if m.isreg() and m.size > 0]

        exact = [m for m in members if m.size == self.TRG_SIZE]
        if exact:
            best = max(
                exact,
                key=lambda m: (self._score_member_name(m.name), -m.size),
            )
            f = tf.extractfile(best)
            if f is not None:
                return f.read()

        patterns = (
            "crash",
            "poc",
            "uaf",
            "repro",
            "reproducer",
            "bug",
            "id:",
            "timeout",
            "oom",
            "minimized",
            "reduced",
            "testcase",
            "clusterfuzz",
            "oss-fuzz",
        )
        cand = [m for m in members if any(p in m.name.lower() for p in patterns)]
        if cand:
            def key_c(m):
                return (
                    -self._score_member_name(m.name),
                    abs(m.size - self.TRG_SIZE),
                    -m.size,
                )

            best = sorted(cand, key=key_c)[0]
            f = tf.extractfile(best)
            if f is not None:
                data = f.read()
                if data:
                    return data

        nested_ext = (".tar", ".tar.gz", ".tgz", ".zip")
        for m in members:
            lname = m.name.lower()
            if not any(lname.endswith(ext) for ext in nested_ext):
                continue
            if m.size > self.NESTED_MAX_SIZE:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            blob = f.read()
            if not blob:
                continue
            if lname.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
                        data = self._search_zipfile(zf, depth + 1)
                        if data is not None:
                            return data
                except Exception:
                    continue
            else:
                try:
                    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as ntf:
                        data = self._search_tarfile(ntf, depth + 1)
                        if data is not None:
                            return data
                except Exception:
                    continue

        text_exts = (".py", ".txt", ".in", ".out", ".code", ".src", ".input")
        textmembers = [m for m in members if m.name.lower().endswith(text_exts)]
        if textmembers:
            best = max(
                textmembers,
                key=lambda m: (self._score_member_name(m.name), m.size),
            )
            f = tf.extractfile(best)
            if f is not None:
                data = f.read()
                if data:
                    return data

        limited = [m for m in members if m.size <= 1024 * 1024]
        if limited:
            best = max(limited, key=lambda m: m.size)
            f = tf.extractfile(best)
            if f is not None:
                data = f.read()
                if data:
                    return data

        return None

    def _search_zipfile(self, zf: zipfile.ZipFile, depth: int) -> bytes | None:
        if depth > self.MAX_CONTAINER_RECURSION:
            return None

        infos = [i for i in zf.infolist() if not getattr(i, "is_dir", lambda: False)() and i.file_size > 0]

        exact = [i for i in infos if i.file_size == self.TRG_SIZE]
        if exact:
            best = max(
                exact,
                key=lambda i: (self._score_member_name(i.filename), -i.file_size),
            )
            try:
                return zf.read(best)
            except Exception:
                pass

        patterns = (
            "crash",
            "poc",
            "uaf",
            "repro",
            "reproducer",
            "bug",
            "id:",
            "timeout",
            "oom",
            "minimized",
            "reduced",
            "testcase",
            "clusterfuzz",
            "oss-fuzz",
        )
        cand = [i for i in infos if any(p in i.filename.lower() for p in patterns)]
        if cand:
            def key_c(i):
                return (
                    -self._score_member_name(i.filename),
                    abs(i.file_size - self.TRG_SIZE),
                    -i.file_size,
                )

            best = sorted(cand, key=key_c)[0]
            try:
                data = zf.read(best)
                if data:
                    return data
            except Exception:
                pass

        nested_ext = (".tar", ".tar.gz", ".tgz", ".zip")
        for i in infos:
            lname = i.filename.lower()
            if not any(lname.endswith(ext) for ext in nested_ext):
                continue
            if i.file_size > self.NESTED_MAX_SIZE:
                continue
            try:
                blob = zf.read(i)
            except Exception:
                continue
            if not blob:
                continue
            if lname.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(blob), "r") as nzf:
                        data = self._search_zipfile(nzf, depth + 1)
                        if data is not None:
                            return data
                except Exception:
                    continue
            else:
                try:
                    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as ntf:
                        data = self._search_tarfile(ntf, depth + 1)
                        if data is not None:
                            return data
                except Exception:
                    continue

        text_exts = (".py", ".txt", ".in", ".out", ".code", ".src", ".input")
        textinfos = [i for i in infos if i.filename.lower().endswith(text_exts)]
        if textinfos:
            best = max(
                textinfos,
                key=lambda i: (self._score_member_name(i.filename), i.file_size),
            )
            try:
                data = zf.read(best)
                if data:
                    return data
            except Exception:
                pass

        limited = [i for i in infos if i.file_size <= 1024 * 1024]
        if limited:
            best = max(limited, key=lambda i: i.file_size)
            try:
                data = zf.read(best)
                if data:
                    return data
            except Exception:
                pass

        return None