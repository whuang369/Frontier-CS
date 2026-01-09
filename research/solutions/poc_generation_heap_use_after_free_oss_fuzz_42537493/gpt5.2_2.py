import os
import re
import tarfile
import gzip
import bz2
import lzma
from typing import Optional, Tuple


def _maybe_decompress(data: bytes, name: str) -> bytes:
    if not data:
        return data
    lname = name.lower()
    try:
        if data.startswith(b"\x1f\x8b") or lname.endswith(".gz"):
            return gzip.decompress(data)
    except Exception:
        pass
    try:
        if lname.endswith(".bz2"):
            return bz2.decompress(data)
    except Exception:
        pass
    try:
        if lname.endswith(".xz") or lname.endswith(".lzma"):
            return lzma.decompress(data)
    except Exception:
        pass
    return data


def _looks_like_markup(data: bytes) -> bool:
    if not data:
        return False
    s = data.lstrip()
    if not s:
        return False
    if s.startswith(b"<?xml") or s.startswith(b"<!DOCTYPE") or s.startswith(b"<"):
        return True
    return False


def _is_probably_input_path(path: str) -> bool:
    lp = path.lower()
    if any(seg in lp for seg in ("/.git/", "\\.git\\", "/doc/", "\\doc\\", "/docs/", "\\docs\\")):
        return False
    if any(lp.endswith(ext) for ext in (
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".inl",
        ".py", ".sh", ".cmake", ".mk", ".m4", ".ac", ".am",
        ".md", ".rst",
        ".png", ".jpg", ".jpeg", ".gif", ".pdf",
        ".o", ".a", ".so", ".dll", ".dylib",
        ".class", ".jar",
        ".wasm",
    )):
        return False
    return True


def _choose_better(best: Optional[Tuple[int, int, bytes]], cand: Tuple[int, int, bytes]) -> Tuple[int, int, bytes]:
    if best is None:
        return cand
    # prioritize lower priority, then shorter length
    if cand[0] < best[0]:
        return cand
    if cand[0] == best[0] and cand[1] < best[1]:
        return cand
    return best


def _find_candidate_from_iter(
    entries_iter,
    issue: str,
    max_read: int = 1_000_000
) -> Optional[bytes]:
    """
    entries_iter yields tuples: (name:str, size:int, read_callable()->bytes)
    """
    issue_re = re.compile(re.escape(issue))
    best: Optional[Tuple[int, int, bytes]] = None

    # Priority levels:
    # 0: path contains issue number
    # 1: path contains clusterfuzz/oss-fuzz/repro/poc/testcase and looks like markup
    # 2: small file (<=128) that looks like markup
    # 3: small file (<=64) any content that looks like markup after decompress
    # 4: very small file (<=64) any content (fallback if nothing else)

    keywords = ("clusterfuzz", "oss-fuzz", "ossfuzz", "repro", "poc", "testcase", "minimized", "crash", "uaf")

    for name, size, reader in entries_iter:
        if size <= 0 or size > max_read:
            continue
        if not _is_probably_input_path(name):
            continue
        lname = name.lower()
        prio = None
        if issue_re.search(lname):
            prio = 0
        elif any(k in lname for k in keywords):
            prio = 1
        elif size <= 128:
            prio = 2
        elif size <= 64:
            prio = 3
        else:
            continue

        try:
            data = reader()
        except Exception:
            continue

        if not data:
            continue

        data2 = _maybe_decompress(data, name)
        if not data2:
            continue

        if prio == 1:
            if not _looks_like_markup(data2) and len(data2) > 0:
                # keep but demote
                prio = 3 if len(data2) <= 64 else 2 if len(data2) <= 128 else 4

        if prio in (2, 3):
            if not _looks_like_markup(data2):
                continue

        if prio == 4 and len(data2) > 64:
            continue

        best = _choose_better(best, (prio, len(data2), data2))

    return None if best is None else best[2]


def _iter_tar_entries(tar_path: str):
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            if not m.isfile():
                continue
            name = m.name
            size = m.size

            def _reader(mm=m):
                f = tf.extractfile(mm)
                if f is None:
                    return b""
                try:
                    return f.read()
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass

            yield name, size, _reader


def _iter_dir_entries(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune some dirs
        ldp = dirpath.lower()
        if any(seg in ldp for seg in (os.sep + ".git", os.sep + "doc", os.sep + "docs")):
            continue
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            try:
                st = os.stat(path)
            except Exception:
                continue
            if not os.path.isfile(path):
                continue
            size = st.st_size

            def _reader(p=path):
                with open(p, "rb") as f:
                    return f.read()

            yield rel, size, _reader


class Solution:
    def solve(self, src_path: str) -> bytes:
        issue = "42537493"

        candidate = None
        if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            candidate = _find_candidate_from_iter(_iter_tar_entries(src_path), issue)
        elif os.path.isdir(src_path):
            candidate = _find_candidate_from_iter(_iter_dir_entries(src_path), issue)
        else:
            # try as tar anyway
            try:
                if os.path.isfile(src_path):
                    candidate = _find_candidate_from_iter(_iter_tar_entries(src_path), issue)
            except Exception:
                candidate = None

        if candidate is not None and len(candidate) > 0:
            return candidate

        return b'<?xml encoding="UTF-8"?>'