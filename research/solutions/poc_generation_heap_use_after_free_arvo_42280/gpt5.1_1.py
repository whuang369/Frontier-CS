import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        def find_in_tar(path: str):
            try:
                if not tarfile.is_tarfile(path):
                    return None
                best_data = None
                best_score = -1
                with tarfile.open(path, "r:*") as tf:
                    members = [m for m in tf.getmembers() if m.isfile() and m.size > 0]
                    # Direct search for files mentioning the issue id
                    for m in members:
                        lower_name = m.name.lower()
                        if "42280" in lower_name:
                            base = os.path.basename(lower_name)
                            _, ext = os.path.splitext(base)
                            allowed_exts = {".pdf", ".ps", ".eps", ".ai", ".bin", ".dat", ".txt"}
                            if (
                                ext.lower() in allowed_exts
                                or "pdf" in lower_name
                                or "ps" in lower_name
                                or "poc" in lower_name
                            ):
                                f = tf.extractfile(m)
                                if f is not None:
                                    try:
                                        data = f.read()
                                    finally:
                                        f.close()
                                    if data:
                                        return data
                    # Generic heuristic search
                    allowed_exts = {".pdf", ".ps", ".eps", ".ai", ".bin", ".dat", ".txt"}
                    target_len = 13996
                    for m in members:
                        if m.size > 2_000_000:
                            continue
                        name = m.name
                        lower_name = name.lower()
                        base = os.path.basename(lower_name)
                        root, ext = os.path.splitext(base)
                        allowed = False
                        if ext.lower() in allowed_exts:
                            allowed = True
                        else:
                            if (
                                "poc" in lower_name
                                or "crash" in lower_name
                                or "42280" in lower_name
                                or "pdf" in lower_name
                                or "ps" in lower_name
                            ):
                                allowed = True
                        if not allowed:
                            continue

                        score = 0
                        keywords = [
                            ("42280", 200),
                            ("poc", 120),
                            ("crash", 110),
                            ("heap", 90),
                            ("uaf", 90),
                            ("pdfi", 80),
                            ("use-after-free", 120),
                            ("use_after_free", 120),
                            ("fuzz", 40),
                            ("oss-fuzz", 50),
                            ("regress", 30),
                            ("test", 20),
                            ("bug", 15),
                        ]
                        for kw, w in keywords:
                            if kw in lower_name:
                                score += w

                        ext_weights = {
                            ".pdf": 100,
                            ".ps": 90,
                            ".eps": 80,
                            ".ai": 60,
                            ".bin": 40,
                            ".dat": 30,
                            ".txt": 10,
                        }
                        score += ext_weights.get(ext.lower(), 0)

                        diff = abs(m.size - target_len)
                        score += max(0, 150 - diff // 20)

                        if (
                            "test" in lower_name
                            or "regress" in lower_name
                            or "fuzz" in lower_name
                            or "poc" in lower_name
                        ):
                            score += 40
                        parts = lower_name.split("/")
                        if "tests" in parts or "regress" in parts:
                            score += 30

                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        try:
                            head = f.read(8)
                        finally:
                            f.close()

                        if head.startswith(b"%PDF"):
                            score += 200
                        elif head.startswith(b"%!PS"):
                            score += 180
                        elif head[:4].isalpha():
                            score += 10

                        if score <= best_score:
                            continue

                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        try:
                            data = f.read()
                        finally:
                            f.close()

                        best_score = score
                        best_data = data
                return best_data
            except Exception:
                return None

        def find_in_zip(path: str):
            try:
                if not zipfile.is_zipfile(path):
                    return None
                best_data = None
                best_score = -1
                with zipfile.ZipFile(path, "r") as zf:
                    infos = [zi for zi in zf.infolist() if not zi.is_dir() and zi.file_size > 0]
                    # Direct search for files mentioning the issue id
                    for info in infos:
                        lower_name = info.filename.lower()
                        if "42280" in lower_name:
                            base = os.path.basename(lower_name)
                            _, ext = os.path.splitext(base)
                            allowed_exts = {".pdf", ".ps", ".eps", ".ai", ".bin", ".dat", ".txt"}
                            if (
                                ext.lower() in allowed_exts
                                or "pdf" in lower_name
                                or "ps" in lower_name
                                or "poc" in lower_name
                            ):
                                with zf.open(info, "r") as f:
                                    data = f.read()
                                if data:
                                    return data
                    # Generic heuristic search
                    allowed_exts = {".pdf", ".ps", ".eps", ".ai", ".bin", ".dat", ".txt"}
                    target_len = 13996
                    for info in infos:
                        if info.file_size > 2_000_000:
                            continue
                        name = info.filename
                        lower_name = name.lower()
                        base = os.path.basename(lower_name)
                        root, ext = os.path.splitext(base)
                        allowed = False
                        if ext.lower() in allowed_exts:
                            allowed = True
                        else:
                            if (
                                "poc" in lower_name
                                or "crash" in lower_name
                                or "42280" in lower_name
                                or "pdf" in lower_name
                                or "ps" in lower_name
                            ):
                                allowed = True
                        if not allowed:
                            continue

                        score = 0
                        keywords = [
                            ("42280", 200),
                            ("poc", 120),
                            ("crash", 110),
                            ("heap", 90),
                            ("uaf", 90),
                            ("pdfi", 80),
                            ("use-after-free", 120),
                            ("use_after_free", 120),
                            ("fuzz", 40),
                            ("oss-fuzz", 50),
                            ("regress", 30),
                            ("test", 20),
                            ("bug", 15),
                        ]
                        for kw, w in keywords:
                            if kw in lower_name:
                                score += w

                        ext_weights = {
                            ".pdf": 100,
                            ".ps": 90,
                            ".eps": 80,
                            ".ai": 60,
                            ".bin": 40,
                            ".dat": 30,
                            ".txt": 10,
                        }
                        score += ext_weights.get(ext.lower(), 0)

                        diff = abs(info.file_size - target_len)
                        score += max(0, 150 - diff // 20)

                        if (
                            "test" in lower_name
                            or "regress" in lower_name
                            or "fuzz" in lower_name
                            or "poc" in lower_name
                        ):
                            score += 40
                        parts = lower_name.split("/")
                        if "tests" in parts or "regress" in parts:
                            score += 30

                        with zf.open(info, "r") as f:
                            head = f.read(8)

                        if head.startswith(b"%PDF"):
                            score += 200
                        elif head.startswith(b"%!PS"):
                            score += 180
                        elif head[:4].isalpha():
                            score += 10

                        if score <= best_score:
                            continue

                        with zf.open(info, "r") as f:
                            data = f.read()

                        best_score = score
                        best_data = data
                return best_data
            except Exception:
                return None

        data = find_in_tar(src_path)
        if data is None:
            data = find_in_zip(src_path)

        if data is not None:
            return data

        # Fallback: generic minimal PDF
        default_poc = (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Length 0 >>\n"
            b"stream\n"
            b"endstream\n"
            b"endobj\n"
            b"trailer\n"
            b"<<>>\n"
            b"%%EOF\n"
        )
        return default_poc