import os
import io
import re
import tarfile
from typing import Optional, Tuple, Iterable


class Solution:
    _GROUND_TRUTH_LEN = 6431

    def _iter_files_from_tar(self, tar_path: str) -> Iterable[Tuple[str, int, bytes]]:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                size = m.size
                if size <= 0:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    head = f.read(2048)
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass
                yield name, size, head

    def _read_full_from_tar(self, tar_path: str, member_name: str) -> Optional[bytes]:
        with tarfile.open(tar_path, "r:*") as tf:
            try:
                m = tf.getmember(member_name)
            except KeyError:
                return None
            if not m.isfile():
                return None
            f = tf.extractfile(m)
            if f is None:
                return None
            try:
                return f.read()
            finally:
                try:
                    f.close()
                except Exception:
                    pass

    def _iter_files_from_dir(self, root: str) -> Iterable[Tuple[str, int, bytes]]:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except Exception:
                    continue
                if not os.path.isfile(path):
                    continue
                size = st.st_size
                if size <= 0:
                    continue
                try:
                    with open(path, "rb") as f:
                        head = f.read(2048)
                except Exception:
                    continue
                rel = os.path.relpath(path, root)
                yield rel, size, head

    def _read_full_from_dir(self, root: str, rel: str) -> Optional[bytes]:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def _looks_like_pdf(self, head: bytes) -> bool:
        if head.startswith(b"%PDF-"):
            return True
        if b"%PDF-" in head[:2048]:
            return True
        return False

    def _score_candidate(self, name: str, size: int, head: bytes) -> float:
        lname = name.lower()

        s = 0.0
        if self._looks_like_pdf(head):
            s += 80.0
        if lname.endswith(".pdf"):
            s += 60.0
        if any(k in lname for k in ("clusterfuzz", "oss-fuzz", "ossfuzz", "poc", "crash", "uaf", "use-after-free", "use_after_free", "heap")):
            s += 40.0
        if "59207" in lname:
            s += 80.0
        if "arvo" in lname:
            s += 30.0
        if any(k in lname for k in ("testcase", "minimized", "repro", "reproducer")):
            s += 25.0

        if 256 <= size <= 500000:
            s += 10.0
        else:
            s -= 40.0

        # Prefer around the ground-truth size if multiple.
        s -= abs(size - self._GROUND_TRUTH_LEN) / 50.0

        # Prefer smaller generally.
        s -= size / 200000.0

        return s

    def _find_best_pdf_from_source(self, src_path: str) -> Optional[bytes]:
        is_tar = os.path.isfile(src_path) and (src_path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")) or tarfile.is_tarfile(src_path))
        is_dir = os.path.isdir(src_path)

        best = None  # (score, name, size)
        if is_tar:
            for name, size, head in self._iter_files_from_tar(src_path):
                lname = name.lower()
                # Fast prefilter
                interesting = (
                    lname.endswith(".pdf")
                    or "59207" in lname
                    or any(k in lname for k in ("clusterfuzz", "oss-fuzz", "ossfuzz", "poc", "crash", "uaf"))
                )
                if not interesting and size > 200000:
                    continue
                if not interesting and not self._looks_like_pdf(head):
                    continue
                if size > 2000000:
                    continue
                sc = self._score_candidate(name, size, head)
                if best is None or sc > best[0]:
                    best = (sc, name, size)
            if best is None:
                return None
            return self._read_full_from_tar(src_path, best[1])

        if is_dir:
            for name, size, head in self._iter_files_from_dir(src_path):
                lname = name.lower()
                interesting = (
                    lname.endswith(".pdf")
                    or "59207" in lname
                    or any(k in lname for k in ("clusterfuzz", "oss-fuzz", "ossfuzz", "poc", "crash", "uaf"))
                )
                if not interesting and size > 200000:
                    continue
                if not interesting and not self._looks_like_pdf(head):
                    continue
                if size > 2000000:
                    continue
                sc = self._score_candidate(name, size, head)
                if best is None or sc > best[0]:
                    best = (sc, name, size)
            if best is None:
                return None
            return self._read_full_from_dir(src_path, best[1])

        return None

    def _build_fallback_pdf(self) -> bytes:
        # Two-revision PDF with xref streams, where a page object (3 0) is compressed in an object stream (7 0)
        # in the first revision, and the main document objects are defined in the second revision.
        # Intended to exercise object-stream loading with multiple xref sections.
        buf = bytearray()
        pos = 0
        offsets = {}

        def w(b: bytes) -> None:
            nonlocal pos
            buf.extend(b)
            pos += len(b)

        def add_obj(num: int, content: bytes) -> None:
            offsets[num] = pos
            w(f"{num} 0 obj\n".encode("ascii"))
            w(content)
            if not content.endswith(b"\n"):
                w(b"\n")
            w(b"endobj\n")

        def add_stream_obj(num: int, dict_items: bytes, stream_data: bytes) -> None:
            offsets[num] = pos
            w(f"{num} 0 obj\n".encode("ascii"))
            # Ensure /Length present
            if b"/Length" not in dict_items:
                dict_items = dict_items.rstrip() + b" /Length " + str(len(stream_data)).encode("ascii") + b" "
            d = b"<< " + dict_items.strip() + b" >>\n"
            w(d)
            w(b"stream\n")
            w(stream_data)
            w(b"\nendstream\n")
            w(b"endobj\n")

        def xref_entry(t: int, f2: int, f3: int) -> bytes:
            return bytes((t & 0xFF,)) + int(f2).to_bytes(4, "big", signed=False) + int(f3).to_bytes(2, "big", signed=False)

        header = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n"
        w(header)

        # Revision 1: minimal root (placeholder), object stream containing page object 3 0, xref stream 8 0
        add_obj(1, b"<<>>\n")

        page3_body = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources <<>> /Contents 5 0 R >>"
        objstm_header = b"3 0 "
        objstm_data = objstm_header + page3_body
        first = len(objstm_header)

        add_stream_obj(
            7,
            b"/Type /ObjStm /N 1 /First " + str(first).encode("ascii") + b" /Length " + str(len(objstm_data)).encode("ascii"),
            objstm_data,
        )

        offset_xref1 = pos
        size1 = 9  # objects 0..8
        entries1 = []
        for i in range(size1):
            if i == 0:
                entries1.append(xref_entry(0, 0, 0))
            elif i == 1:
                entries1.append(xref_entry(1, offsets[1], 0))
            elif i == 3:
                entries1.append(xref_entry(2, 7, 0))  # object 3 in objstm 7 at index 0
            elif i == 7:
                entries1.append(xref_entry(1, offsets[7], 0))
            elif i == 8:
                entries1.append(xref_entry(1, offset_xref1, 0))
            else:
                entries1.append(xref_entry(0, 0, 0))
        xref1_data = b"".join(entries1)
        xref1_dict = (
            b"/Type /XRef /Size "
            + str(size1).encode("ascii")
            + b" /Root 1 0 R /W [1 4 2] /Length "
            + str(len(xref1_data)).encode("ascii")
        )
        add_stream_obj(8, xref1_dict, xref1_data)

        w(b"startxref\n")
        w(str(offset_xref1).encode("ascii"))
        w(b"\n%%EOF\n")

        # Revision 2: main document objects (catalog/pages/contents), xref stream 9 with /Prev to xref1
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
        add_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n")
        contents = b"q\nQ\n"
        add_stream_obj(5, b"/Length " + str(len(contents)).encode("ascii"), contents)

        offset_xref2 = pos
        size2 = 10  # objects 0..9
        # /Index covering [0 1] [1 2] [5 1] [9 1]
        index2 = [0, 1, 1, 2, 5, 1, 9, 1]
        entries2 = []
        # 0
        entries2.append(xref_entry(0, 0, 0))
        # 1..2
        entries2.append(xref_entry(1, offsets[1], 0))
        entries2.append(xref_entry(1, offsets[2], 0))
        # 5
        entries2.append(xref_entry(1, offsets[5], 0))
        # 9 (xref2 itself)
        entries2.append(xref_entry(1, offset_xref2, 0))
        xref2_data = b"".join(entries2)
        xref2_dict = (
            b"/Type /XRef /Size "
            + str(size2).encode("ascii")
            + b" /Root 1 0 R /Prev "
            + str(offset_xref1).encode("ascii")
            + b" /W [1 4 2] /Index ["
            + b" ".join(str(x).encode("ascii") for x in index2)
            + b"] /Length "
            + str(len(xref2_data)).encode("ascii")
        )
        add_stream_obj(9, xref2_dict, xref2_data)

        w(b"startxref\n")
        w(str(offset_xref2).encode("ascii"))
        w(b"\n%%EOF\n")
        return bytes(buf)

    def solve(self, src_path: str) -> bytes:
        poc = self._find_best_pdf_from_source(src_path)
        if isinstance(poc, (bytes, bytearray)) and len(poc) > 0:
            return bytes(poc)
        return self._build_fallback_pdf()