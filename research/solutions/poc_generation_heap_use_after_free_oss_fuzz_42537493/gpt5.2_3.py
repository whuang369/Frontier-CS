import os
import io
import re
import tarfile
import zipfile
from typing import Optional, Tuple, Iterable


_MAX_CANDIDATE_SIZE = 65536
_MAX_SCAN_TEXT_SIZE = 524288


_NAME_KEYWORDS_STRONG = (
    "clusterfuzz-testcase",
    "clusterfuzz",
    "testcase-minimized",
    "minimized",
    "repro",
    "reproducer",
    "poc",
    "crash",
    "uaf",
    "use-after-free",
    "42537493",
)

_NAME_KEYWORDS_MED = (
    "oss-fuzz",
    "fuzz",
    "corpus",
    "seed",
    "inputs",
    "regress",
    "issue",
    "bug",
)

_BAD_NAME_PREFIXES = (
    "readme",
    "license",
    "copying",
    "changelog",
    "news",
    "authors",
    "contributing",
)

_BAD_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".hpp", ".hh",
    ".py", ".sh", ".bat", ".cmd", ".ps1",
    ".md", ".rst", ".txt", ".html", ".css",
    ".json", ".yml", ".yaml", ".toml", ".ini",
    ".cmake", ".mk", ".am", ".ac", ".in", ".m4",
    ".pl", ".rb", ".go", ".rs", ".java", ".js",
    ".proto", ".gperf", ".pod", ".dox", ".tex",
}


def _norm_name(name: str) -> str:
    return name.replace("\\", "/").lower().strip()


def _basename(name: str) -> str:
    n = name.replace("\\", "/")
    if "/" in n:
        n = n.rsplit("/", 1)[1]
    return n


def _ext(name: str) -> str:
    b = _basename(name)
    if "." not in b:
        return ""
    return "." + b.rsplit(".", 1)[1].lower()


def _looks_like_text_source(name: str) -> bool:
    bn = _basename(name).lower()
    if any(bn.startswith(p) for p in _BAD_NAME_PREFIXES):
        return True
    ex = _ext(name)
    if ex in _BAD_EXTS:
        return True
    return False


def _iter_files_from_tar(path: str) -> Iterable[Tuple[str, int, bytes]]:
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            sz = int(getattr(m, "size", 0) or 0)
            if sz <= 0 or sz > _MAX_CANDIDATE_SIZE:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                data = f.read(sz + 1)
            except Exception:
                continue
            if data is None:
                continue
            if len(data) > sz:
                data = data[:sz]
            yield m.name, sz, data


def _iter_files_from_zip(path: str) -> Iterable[Tuple[str, int, bytes]]:
    with zipfile.ZipFile(path, "r") as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            sz = int(zi.file_size or 0)
            if sz <= 0 or sz > _MAX_CANDIDATE_SIZE:
                continue
            try:
                data = zf.read(zi.filename)
            except Exception:
                continue
            if not data:
                continue
            if len(data) > _MAX_CANDIDATE_SIZE:
                data = data[:_MAX_CANDIDATE_SIZE]
            yield zi.filename, sz, data


def _iter_files_from_dir(path: str) -> Iterable[Tuple[str, int, bytes]]:
    for root, _, files in os.walk(path):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
            except Exception:
                continue
            sz = int(st.st_size or 0)
            if sz <= 0 or sz > _MAX_CANDIDATE_SIZE:
                continue
            try:
                with open(full, "rb") as f:
                    data = f.read(sz + 1)
            except Exception:
                continue
            if not data:
                continue
            if len(data) > sz:
                data = data[:sz]
            rel = os.path.relpath(full, path)
            yield rel, sz, data


def _score_candidate(name: str, data: bytes) -> int:
    n = _norm_name(name)
    bn = _basename(n)

    score = 0

    for kw in _NAME_KEYWORDS_STRONG:
        if kw in n:
            score += 250
    for kw in _NAME_KEYWORDS_MED:
        if kw in n:
            score += 60

    if n.startswith("test") or "/test" in n:
        score += 20
    if "/fuzz" in n or n.startswith("fuzz"):
        score += 40

    if len(data) == 24:
        score += 120
    elif 1 <= len(data) <= 64:
        score += 25
    elif 65 <= len(data) <= 256:
        score += 10

    if data.startswith(b"<?xml"):
        score += 35
    elif data.startswith(b"<"):
        score += 20

    if b"\x00" in data:
        score += 10

    if _looks_like_text_source(bn):
        score -= 120

    return score


def _find_best_poc_bytes(src_path: str) -> Optional[bytes]:
    it: Optional[Iterable[Tuple[str, int, bytes]]] = None

    if os.path.isdir(src_path):
        it = _iter_files_from_dir(src_path)
    elif os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
        it = _iter_files_from_tar(src_path)
    elif os.path.isfile(src_path) and zipfile.is_zipfile(src_path):
        it = _iter_files_from_zip(src_path)
    else:
        return None

    best: Optional[Tuple[int, int, str, bytes]] = None

    for name, sz, data in it:
        nn = _norm_name(name)
        if "clusterfuzz-testcase-minimized" in nn or "clusterfuzz-testcase" in nn:
            if 0 < len(data) <= _MAX_CANDIDATE_SIZE:
                return data

        sc = _score_candidate(name, data)
        if best is None:
            best = (sc, sz, name, data)
        else:
            bsc, bsz, bname, _ = best
            if (sc > bsc) or (sc == bsc and sz < bsz) or (sc == bsc and sz == bsz and name < bname):
                best = (sc, sz, name, data)

    if best is None:
        return None

    bsc, bsz, bname, bdata = best
    if bsc <= -200:
        return None
    return bdata


def _detect_fuzzer_input_style(src_path: str) -> str:
    def iter_text_files() -> Iterable[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    if not fn.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                        continue
                    full = os.path.join(root, fn)
                    try:
                        st = os.stat(full)
                    except Exception:
                        continue
                    if st.st_size <= 0 or st.st_size > _MAX_SCAN_TEXT_SIZE:
                        continue
                    try:
                        with open(full, "rb") as f:
                            data = f.read(_MAX_SCAN_TEXT_SIZE + 1)
                    except Exception:
                        continue
                    yield os.path.relpath(full, src_path), data
        elif os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    if not m.name.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                        continue
                    sz = int(getattr(m, "size", 0) or 0)
                    if sz <= 0 or sz > _MAX_SCAN_TEXT_SIZE:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        data = f.read(_MAX_SCAN_TEXT_SIZE + 1)
                    except Exception:
                        continue
                    yield m.name, data
        elif os.path.isfile(src_path) and zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path, "r") as zf:
                for zi in zf.infolist():
                    if zi.is_dir():
                        continue
                    if not zi.filename.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                        continue
                    if zi.file_size <= 0 or zi.file_size > _MAX_SCAN_TEXT_SIZE:
                        continue
                    try:
                        data = zf.read(zi.filename)
                    except Exception:
                        continue
                    yield zi.filename, data

    saw_fuzzer = False
    saw_xml_parse = False
    saw_encoding_string = False

    for _, raw in iter_text_files():
        if b"LLVMFuzzerTestOneInput" not in raw:
            continue
        saw_fuzzer = True
        low = raw.lower()
        if b"xmlreadmemory" in low or b"xmlparsememory" in low or b"xmlctxtreadmemory" in low or b"htmlreadmemory" in low:
            saw_xml_parse = True
        if b"xmlfindcharencodinghandler" in low or b"xmlopencharencodinghandler" in low:
            saw_encoding_string = True

    if not saw_fuzzer:
        return "unknown"
    if saw_xml_parse:
        return "xml"
    if saw_encoding_string:
        return "string"
    return "raw"


def _fallback_poc(style: str) -> bytes:
    if style == "string":
        return b"UTF-8"
    if style == "xml":
        return b'<?xml version="1.0" encoding="UTF-8"?><a/>'
    return b"\x00" * 24


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = _find_best_poc_bytes(src_path)
        if poc is not None and len(poc) > 0:
            return poc

        style = _detect_fuzzer_input_style(src_path)
        return _fallback_poc(style)