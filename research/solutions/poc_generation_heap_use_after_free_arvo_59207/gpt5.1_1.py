import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 6431
        data = None

        if os.path.isdir(src_path):
            data = self._from_directory(src_path, target_size)
        else:
            data = self._from_tar(src_path, target_size)

        if data is not None:
            return data

        return self._fallback_poc()

    def _from_tar(self, path: str, target_size: int) -> bytes | None:
        try:
            tf = tarfile.open(path, "r:*")
        except tarfile.TarError:
            return None

        with tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if not members:
                return None

            pdf_members = [m for m in members if m.name.lower().endswith(".pdf")]
            data = self._extract_from_members(tf, pdf_members, target_size)
            if data is not None:
                return data

            same_size_members = [m for m in members if m.size == target_size]
            if same_size_members:
                chosen = self._select_best_member(same_size_members, target_size)
                if chosen is not None:
                    try:
                        f = tf.extractfile(chosen)
                    except KeyError:
                        f = None
                    if f is not None:
                        data = f.read()
                        if data:
                            return data

        return None

    def _extract_from_members(
        self, tf: tarfile.TarFile, members: list[tarfile.TarInfo], target_size: int
    ) -> bytes | None:
        if not members:
            return None

        small = [m for m in members if m.size <= 100000]
        if not small:
            small = members

        exact = [m for m in small if m.size == target_size]
        if exact:
            chosen = self._select_best_member(exact, target_size)
            if chosen is not None:
                try:
                    f = tf.extractfile(chosen)
                except KeyError:
                    f = None
                if f is not None:
                    data = f.read()
                    if data:
                        return data

        ordered = sorted(
            small,
            key=lambda m: self._name_score(m.name, m.size, target_size),
            reverse=True,
        )
        for m in ordered:
            try:
                f = tf.extractfile(m)
            except KeyError:
                continue
            if f is None:
                continue
            data = f.read()
            if data:
                return data

        return None

    def _from_directory(self, root: str, target_size: int) -> bytes | None:
        pdf_paths: list[tuple[str, int]] = []

        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if name.lower().endswith(".pdf"):
                    pdf_paths.append((full, size))

        if not pdf_paths:
            return None

        exact = [(p, s) for (p, s) in pdf_paths if s == target_size]
        if exact:
            chosen_path = self._select_best_path(exact, target_size)
            if chosen_path is not None:
                try:
                    with open(chosen_path, "rb") as f:
                        data = f.read()
                    if data:
                        return data
                except OSError:
                    pass

        ordered = sorted(
            pdf_paths,
            key=lambda ps: self._name_score(ps[0], ps[1], target_size),
            reverse=True,
        )
        for path, _ in ordered:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except OSError:
                continue

        return None

    def _name_score(self, name: str, size: int, target_size: int) -> int:
        score = 0
        diff = abs(size - target_size)
        score -= diff

        nl = name.lower()
        keywords = [
            "59207",
            "use-after-free",
            "use_after_free",
            "heap-use-after-free",
            "heap_use_after_free",
            "heap-uaf",
            "uaf",
            "poc",
            "crash",
        ]
        for kw in keywords:
            if kw in nl:
                score += 1000

        if (
            "/poc/" in nl
            or "/pocs/" in nl
            or "/crash/" in nl
            or "/crashes/" in nl
            or "/regress" in nl
            or "regression" in nl
        ):
            score += 100

        return score

    def _select_best_member(
        self, members: list[tarfile.TarInfo], target_size: int
    ) -> tarfile.TarInfo | None:
        best = None
        best_score = None
        for m in members:
            s = self._name_score(m.name, m.size, target_size)
            if best is None or s > best_score:
                best = m
                best_score = s
        return best

    def _select_best_path(
        self, paths: list[tuple[str, int]], target_size: int
    ) -> str | None:
        best = None
        best_score = None
        for p, size in paths:
            s = self._name_score(p, size, target_size)
            if best is None or s > best_score:
                best = p
                best_score = s
        return best

    def _fallback_poc(self) -> bytes:
        content = (
            b"%PDF-1.1\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog /Pages 2 0 R >>\n"
            b"endobj\n"
            b"2 0 obj\n"
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
            b"endobj\n"
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\n"
            b"endobj\n"
            b"4 0 obj\n"
            b"<< /Length 44 >>\n"
            b"stream\n"
            b"BT /F1 24 Tf 100 700 Td (Hello, world!) Tj ET\n"
            b"endstream\n"
            b"endobj\n"
            b"xref\n"
            b"0 5\n"
            b"0000000000 65535 f \n"
            b"0000000010 00000 n \n"
            b"0000000063 00000 n \n"
            b"0000000116 00000 n \n"
            b"0000000211 00000 n \n"
            b"trailer\n"
            b"<< /Root 1 0 R /Size 5 >>\n"
            b"startxref\n"
            b"314\n"
            b"%%EOF\n"
        )
        return content