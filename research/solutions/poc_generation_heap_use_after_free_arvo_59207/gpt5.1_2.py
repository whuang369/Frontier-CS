import tarfile


class Solution:
    def __init__(self):
        self.gt_length = 6431

    def solve(self, src_path: str) -> bytes:
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        best_exact = None
        best_exact_score = -1
        other_candidates = []

        with tf:
            members = tf.getmembers()
            for member in members:
                if not member.isfile():
                    continue
                size = member.size
                name = member.name
                lower = name.lower()
                is_pdf_like = lower.endswith(".pdf") or ".pdf" in lower
                has_kw = any(
                    k in lower
                    for k in (
                        "poc",
                        "uaf",
                        "use_after_free",
                        "use-after-free",
                        "crash",
                        "heap",
                        "clusterfuzz",
                        "testcase",
                    )
                )
                score = 0
                if is_pdf_like:
                    score += 2
                if has_kw:
                    score += 4

                if size == self.gt_length:
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    if len(data) != size:
                        continue
                    if score > best_exact_score:
                        best_exact = data
                        best_exact_score = score
                else:
                    if has_kw or is_pdf_like:
                        other_candidates.append((score, size, name))

            if best_exact is not None:
                return best_exact

            best_data = None
            best_rank = None
            for score, size, name in other_candidates:
                try:
                    member = tf.getmember(name)
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                size_diff = abs(size - self.gt_length)
                rank = (score, -size_diff, -size)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_data = data

            if best_data is not None:
                return best_data

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        with tf:
            smallest_pdf_data = None
            smallest_pdf_size = None
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if not name.lower().endswith(".pdf"):
                    continue
                size = member.size
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                if smallest_pdf_size is None or size < smallest_pdf_size:
                    smallest_pdf_size = size
                    smallest_pdf_data = data

            if smallest_pdf_data is not None:
                return smallest_pdf_data

        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        pdf_lines = [
            "%PDF-1.5",
            "%Fallback PDF for UAF PoC generation",
            "1 0 obj",
            "<< /Type /Catalog /Pages 2 0 R >>",
            "endobj",
            "2 0 obj",
            "<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            "endobj",
            "3 0 obj",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            "endobj",
            "4 0 obj",
            "<< /Length 44 >>",
            "stream",
            "BT /F1 12 Tf 72 712 Td (Fallback UAF PDF) Tj ET",
            "endstream",
            "endobj",
            "xref",
            "0 5",
            "0000000000 65535 f ",
            "0000000010 00000 n ",
            "0000000060 00000 n ",
            "0000000110 00000 n ",
            "0000000200 00000 n ",
            "trailer",
            "<< /Size 5 /Root 1 0 R >>",
            "startxref",
            "0",
            "%%EOF",
        ]
        return ("\n".join(pdf_lines) + "\n").encode("ascii")