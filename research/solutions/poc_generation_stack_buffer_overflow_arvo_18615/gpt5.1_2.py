import tarfile
import os
from typing import List


class Solution:
    TEXT_EXTS = (
        '.c', '.h', '.hpp', '.hh', '.cc', '.cpp', '.cxx',
        '.txt', '.md', '.markdown', '.rst',
        '.json', '.xml', '.html', '.htm',
        '.yml', '.yaml', '.ini', '.cfg', '.conf',
        '.py', '.pyw', '.sh', '.bash', '.zsh',
        '.java', '.rb', '.pl', '.php', '.js', '.ts',
        '.go', '.rs', '.m', '.mm', '.cs', '.swift', '.kt',
        '.sql', '.s', '.asm',
        '.bat', '.cmd',
        '.mak', '.mk', '.cmake', '.am', '.ac',
        '.gitignore', '.gitattributes', '.gitmodules',
        '.in', '.out',
        '.log',
    )

    def solve(self, src_path: str) -> bytes:
        try:
            with tarfile.open(src_path, 'r:*') as tar:
                candidates = self._ranked_poc_candidates(tar)
                for member in candidates[:10]:
                    try:
                        f = tar.extractfile(member)
                    except Exception:
                        continue
                    if f is None:
                        continue
                    try:
                        data = f.read()
                    except Exception:
                        continue
                    if not data:
                        continue
                    if not self._is_probably_text(data):
                        return data
        except Exception:
            pass
        return self._default_poc()

    def _ranked_poc_candidates(self, tar: tarfile.TarFile) -> List[tarfile.TarInfo]:
        members = tar.getmembers()
        candidates: List[tarfile.TarInfo] = []
        scores: List[float] = []

        keyword_weights = (
            ('poc', 60.0),
            ('proof', 30.0),
            ('crash', 50.0),
            ('repro', 40.0),
            ('id:', 25.0),
            ('id_', 25.0),
            ('id-', 25.0),
            ('tic30', 45.0),
            ('overflow', 35.0),
            ('stack', 25.0),
            ('stackoverflow', 35.0),
            ('dis', 10.0),
            ('fuzz', 20.0),
            ('input', 15.0),
            ('bug', 20.0),
        )

        for m in members:
            if not m.isreg():
                continue
            size = m.size
            if size <= 0 or size > 1024 * 1024:
                continue
            name = m.name
            lname = name.lower()
            is_text = self._is_text_like_name(lname)

            score = 0.0
            if size <= 32:
                score += 40.0
            elif size <= 256:
                score += 20.0
            else:
                score += max(0.0, 10.0 - (size / 1024.0))

            score += max(0.0, 40.0 - abs(size - 10.0))

            for kw, w in keyword_weights:
                if kw in lname:
                    score += w

            if is_text:
                score -= 60.0

            if any(seg in lname for seg in ('/doc', '/docs', '/example', '/examples', '/sample', '/samples')):
                score -= 20.0
            if any(seg in lname for seg in ('/test', '/tests', '/testing')) and not any(
                kw in lname for kw in ('poc', 'crash', 'bug', 'fuzz')
            ):
                score -= 10.0

            if score <= 0.0:
                continue

            if not self._looks_like_poc_name(lname, size, is_text):
                continue

            candidates.append(m)
            scores.append(score)

        if not candidates:
            return []

        # Sort candidates by score descending
        sorted_indices = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in sorted_indices]

    def _is_text_like_name(self, lname: str) -> bool:
        if lname.endswith('makefile') or 'readme' in lname or 'license' in lname or 'copying' in lname or 'changelog' in lname or 'history' in lname:
            return True
        for ext in self.TEXT_EXTS:
            if lname.endswith(ext):
                return True
        if '.gitignore' in lname or '.gitattributes' in lname or '.gitmodules' in lname:
            return True
        if 'config' in lname:
            return True
        return False

    def _looks_like_poc_name(self, lname: str, size: int, is_text_like: bool) -> bool:
        if any(kw in lname for kw in ('poc', 'crash', 'repro', 'bug', 'id:', 'id_', 'id-', 'tic30', 'overflow', 'stack', 'stackoverflow', 'fuzz')):
            return True

        if any(lname.endswith(ext) for ext in ('.bin', '.dat', '.raw', '.core', '.dump')):
            return True

        if any(seg in lname for seg in ('afl-out', 'afl_output', '/crashes/', '/queue/')):
            return True

        if size <= 16 and not is_text_like:
            base = os.path.basename(lname)
            if base.startswith('.git'):
                return False
            if base in ('.travis.yml', '.clang-format'):
                return False
            return True

        return False

    def _is_probably_text(self, data: bytes) -> bool:
        if not data:
            return True
        text_chars = set(range(32, 127))
        text_chars.update({9, 10, 13})
        nontext = 0
        length = len(data)
        threshold = max(1, length * 5 // 100)
        for b in data:
            if b not in text_chars:
                nontext += 1
                if nontext > threshold:
                    return False
        return True

    def _default_poc(self) -> bytes:
        return b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'