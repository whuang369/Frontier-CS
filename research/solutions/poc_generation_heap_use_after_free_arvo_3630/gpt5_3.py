import os
import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        proj_str = "+proj=lsat"
        add_lsat = True
        add_path = True

        try:
            if src_path and os.path.exists(src_path) and tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, "r:*") as tf:
                    member = None
                    for m in tf.getmembers():
                        name_lower = m.name.lower()
                        if name_lower.endswith("pj_lsat.c") or name_lower.endswith("pj_lsat.cpp") or re.search(r"(^|/)(pj_|pj)lsat\.(c|cc|cpp)$", name_lower):
                            member = m
                            break
                    if member is not None:
                        f = tf.extractfile(member)
                        if f:
                            try:
                                content = f.read().decode("utf-8", errors="ignore")
                                # Detect parameter names expected by PJ_lsat.c
                                # Common forms: "ilsat", "ipath"
                                add_lsat = ("\"ilsat\"" in content) or ("\"lsat\"" in content)
                                add_path = ("\"ipath\"" in content) or ("\"path\"" in content)
                            except Exception:
                                pass
        except Exception:
            pass

        if add_lsat:
            proj_str += " +lsat=1"
        if add_path:
            proj_str += " +path=1"

        proj_str += "\n"
        return proj_str.encode("ascii", errors="ignore")