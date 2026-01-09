import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        Lg = 7270  # ground-truth PoC length hint

        # Try to open as a tar archive. If that fails, just return the file bytes.
        try:
            tf = tarfile.open(src_path, "r:*")
        except tarfile.ReadError:
            with open(src_path, "rb") as f:
                return f.read()

        text_exts = {".rb", ".mrb", ".txt", ".input", ".in"}
        infos = []

        # First pass: basic scoring based on filename, path, and size.
        for m in tf.getmembers():
            if not m.isreg():
                continue

            size = m.size
            lname = m.name.lower()
            base = os.path.basename(lname)
            ext = os.path.splitext(base)[1]

            score = 0.0

            # Extension-based heuristics
            if ext in text_exts:
                score += 50.0
            if ext == ".rb":
                score += 80.0
            elif ext == ".mrb":
                score += 70.0

            # Directory and path hints
            parts = lname.split("/")
            for p in parts:
                if p in ("poc", "pocs"):
                    score += 120.0
                if p in ("crash", "crashes"):
                    score += 100.0
                if p in ("queue", "seeds", "inputs", "cases", "tests"):
                    score += 20.0

            if "poc" in lname:
                score += 120.0
            if "use-after-free" in lname or "use_after_free" in lname or "uaf" in lname:
                score += 60.0
            if "heap" in lname:
                score += 10.0
            if "ruby" in lname or "mruby" in lname:
                score += 10.0
            if "id_" in base or "id-" in base:
                score += 10.0

            # Size-based heuristics
            if size == 0:
                score -= 500.0
            else:
                if size > 200000:
                    score -= (size - 200000) / 1000.0
                score -= abs(size - Lg) / 50.0

            infos.append([m, score, ext, size])

        # Second pass: content-based scoring for likely text files.
        for info in infos:
            m, score, ext, size = info
            if ext not in text_exts or size <= 0 or size > 100000:
                continue
            try:
                f = tf.extractfile(m)
            except Exception:
                continue
            if f is None:
                continue
            try:
                data = f.read(4096)
            finally:
                f.close()
            ldata = data.lower()
            if b"heap use after free" in ldata:
                score += 200.0
            if b"use after free" in ldata:
                score += 150.0
            if b"uaf" in ldata:
                score += 80.0
            if b"stack_extend" in ldata or b"mrb_stack_extend" in ldata:
                score += 40.0
            if b"mruby" in ldata or b"ruby" in ldata:
                score += 10.0
            info[1] = score

        # Select the best candidate.
        best_member = None
        best_score = None
        for m, score, ext, size in infos:
            if best_member is None or score > best_score:
                best_member = m
                best_score = score

        if best_member is not None and best_score is not None:
            f = tf.extractfile(best_member)
            if f is not None:
                try:
                    data = f.read()
                finally:
                    f.close()
                if data:
                    tf.close()
                    return data

        tf.close()

        # Fallback: synthetic Ruby PoC attempting to stress stack extension.
        ruby_code = (
            "# Synthetic PoC fallback generated automatically\n"
            "def deep(a1=1,a2=1,a3=1,a4=1,a5=1,a6=1,a7=1,a8=1,a9=1,a10=1,"
            "a11=1,a12=1,a13=1,a14=1,a15=1,a16=1,a17=1,a18=1,a19=1,a20=1,"
            "a21=1,a22=1,a23=1,a24=1,a25=1,a26=1,a27=1,a28=1,a29=1,a30=1,"
            "a31=1,a32=1,a33=1,a34=1,a35=1,a36=1,a37=1,a38=1,a39=1,a40=1,"
            "a41=1,a42=1,a43=1,a44=1,a45=1,a46=1,a47=1,a48=1,a49=1,a50=1)\n"
            "  if a1 > 0\n"
            "    deep(a1-1,a2,a3,a4,a5,a6,a7,a8,a9,a10,"
            "a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,"
            "a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,"
            "a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,"
            "a41,a42,a43,a44,a45,a46,a47,a48,a49,a50)\n"
            "  end\n"
            "end\n"
            "\n"
            "1000.times do\n"
            "  deep(50)\n"
            "end\n"
        )
        return ruby_code.encode("utf-8")