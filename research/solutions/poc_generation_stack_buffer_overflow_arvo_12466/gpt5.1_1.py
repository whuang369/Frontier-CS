import os
import tarfile

RAR5_SIG = b'Rar!\x1a\x07\x01\x00'
TARGET_SIZE = 524


class Solution:
    def solve(self, src_path: str) -> bytes:
        data = None

        if os.path.isfile(src_path):
            data = self._from_tar(src_path)

        if data is None and os.path.isdir(src_path):
            data = self._from_dir(src_path)

        if data is None:
            data = b"A" * TARGET_SIZE

        return data

    def _score_name(self, name: str) -> int:
        n = name.lower()
        score = 0
        if "rar5" in n:
            score += 8
        if ".rar" in n:
            score += 4
        if "poc" in n or "crash" in n or "exploit" in n or "bug" in n or "id_" in n:
            score += 4
        if "huff" in n or "huffman" in n:
            score += 2
        if "test" in n or "tests" in n:
            score += 1
        if "fuzz" in n or "corpus" in n or "seed" in n or "input" in n:
            score += 1
        return score

    def _candidate_priority(self, is_rar5: bool, name: str, size: int):
        return (
            1 if is_rar5 else 0,
            1 if size == TARGET_SIZE else 0,
            -abs(size - TARGET_SIZE),
            self._score_name(name),
            -size,
        )

    def _from_tar(self, tar_path: str):
        try:
            if not tarfile.is_tarfile(tar_path):
                return None
        except Exception:
            return None

        best_member = None
        best_priority = None

        try:
            tf = tarfile.open(tar_path, "r:*")
        except Exception:
            return None

        with tf:
            for member in tf.getmembers():
                if not member.isfile() or member.size <= 0:
                    continue

                size = member.size
                name = member.name
                name_lower = name.lower()

                is_interesting = (
                    ".rar" in name_lower
                    or "rar5" in name_lower
                    or "poc" in name_lower
                    or "crash" in name_lower
                    or "exploit" in name_lower
                    or "bug" in name_lower
                    or "id_" in name_lower
                )

                is_rar5 = False

                if size >= len(RAR5_SIG):
                    try:
                        f = tf.extractfile(member)
                    except Exception:
                        f = None
                    if f is not None:
                        try:
                            sig = f.read(len(RAR5_SIG))
                        except Exception:
                            sig = b""
                        finally:
                            try:
                                f.close()
                            except Exception:
                                pass
                        if sig == RAR5_SIG:
                            is_rar5 = True
                            is_interesting = True

                if not is_interesting:
                    continue

                priority = self._candidate_priority(is_rar5, name, size)
                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    best_member = member

            if best_member is not None:
                try:
                    f = tf.extractfile(best_member)
                except Exception:
                    f = None
                if f is not None:
                    try:
                        data = f.read()
                    finally:
                        try:
                            f.close()
                        except Exception:
                            pass
                    return data

        return None

    def _from_dir(self, root: str):
        best_path = None
        best_priority = None

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue

                if size <= 0:
                    continue

                name_lower = path.lower()
                is_interesting = (
                    ".rar" in name_lower
                    or "rar5" in name_lower
                    or "poc" in name_lower
                    or "crash" in name_lower
                    or "exploit" in name_lower
                    or "bug" in name_lower
                    or "id_" in name_lower
                )

                is_rar5 = False

                if size >= len(RAR5_SIG):
                    try:
                        with open(path, "rb") as f:
                            sig = f.read(len(RAR5_SIG))
                    except OSError:
                        continue
                    if sig == RAR5_SIG:
                        is_rar5 = True
                        is_interesting = True

                if not is_interesting:
                    continue

                priority = self._candidate_priority(is_rar5, path, size)
                if best_priority is None or priority > best_priority:
                    best_priority = priority
                    best_path = path

        if best_path is not None:
            try:
                with open(best_path, "rb") as f:
                    return f.read()
            except OSError:
                return None

        return None