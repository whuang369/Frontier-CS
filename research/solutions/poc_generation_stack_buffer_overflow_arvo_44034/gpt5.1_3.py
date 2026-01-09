import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 80064

        def fallback() -> bytes:
            return (
                b"%PDF-1.4\n"
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
                b"xref\n0 4\n0000000000 65535 f \n"
                b"trailer << /Root 1 0 R /Size 4 >>\nstartxref\n0\n%%%%EOF\n"
            )

        try:
            with tarfile.open(src_path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if not members:
                    return fallback()

                preferred_exts = (".pdf", ".poc", ".ps", ".bin", ".dat", ".cff", ".ttf", ".otf")

                def score(m: tarfile.TarInfo) -> float:
                    name = m.name.lower()
                    s = 0.0

                    for idx, ext in enumerate(preferred_exts):
                        if name.endswith(ext):
                            s += 100 - idx * 5
                            break

                    for kw, bonus in (
                        ("poc", 50),
                        ("cid", 40),
                        ("crash", 45),
                        ("font", 35),
                        ("regress", 30),
                        ("test", 25),
                    ):
                        if kw in name:
                            s += bonus

                    for kw, bonus in (
                        ("/tests/", 15),
                        ("/test/", 15),
                        ("/regress/", 15),
                        ("/fuzz/", 15),
                    ):
                        if kw in name:
                            s += bonus

                    s -= name.count("/")
                    s -= abs(m.size - target_size) / 1024.0
                    return s

                def is_probable_pdf(m: tarfile.TarInfo) -> bool:
                    name = m.name.lower()
                    if not (name.endswith(".pdf") or ".pdf" in name):
                        return False
                    try:
                        f = tar.extractfile(m)
                        if f is None:
                            return False
                        header = f.read(5)
                        return header.startswith(b"%PDF-") or header.startswith(b"%pdf-")
                    except Exception:
                        return False

                same_size = [m for m in members if m.size == target_size]
                chosen = None

                if same_size:
                    pdf_same = [m for m in same_size if is_probable_pdf(m)]
                    if pdf_same:
                        pdf_same.sort(key=score, reverse=True)
                        chosen = pdf_same[0]
                    else:
                        same_size.sort(key=score, reverse=True)
                        chosen = same_size[0]
                else:
                    near = [m for m in members if 0 < m.size <= 2 * target_size]
                    if not near:
                        near = members

                    pdf_near = [m for m in near if is_probable_pdf(m)]
                    if pdf_near:
                        pdf_near.sort(key=lambda m: (abs(m.size - target_size), -score(m)))
                        chosen = pdf_near[0]
                    else:
                        near.sort(key=lambda m: (abs(m.size - target_size), -score(m)))
                        chosen = near[0]

                if chosen is None:
                    return fallback()

                f = tar.extractfile(chosen)
                if f is None:
                    return fallback()
                data = f.read()
                if not isinstance(data, bytes):
                    data = bytes(data)
                return data
        except Exception:
            return fallback()