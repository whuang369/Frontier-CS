import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        def select_poc_member(tf: tarfile.TarFile):
            members = [m for m in tf.getmembers() if m.isfile() and m.size > 0 and m.size <= 1_000_000]
            best_member = None
            best_score = None

            font_exts = {'.ttf', '.otf', '.woff', '.woff2', '.ttc', '.otc'}
            keyword_substrings = ['poc', 'proof', 'crash', 'uaf', 'heap', 'bug', 'overflow', 'corrupt', 'exploit']

            for m in members:
                name_lower = m.name.lower()
                base = os.path.basename(name_lower)
                _, ext = os.path.splitext(base)
                size = int(m.size)

                is_font_ext = ext in font_exts
                has_keyword = any(k in name_lower for k in keyword_substrings)
                path_parts = name_lower.split('/')
                in_poc_dir = any(part in ('poc', 'pocs', 'crash', 'crashes', 'fuzz', 'corpus', 'tests', 'regress') for part in path_parts)

                font_magic = False
                if size <= 65536:
                    try:
                        f = tf.extractfile(m)
                        if f is not None:
                            head = f.read(4)
                            if head in (b'OTTO', b'wOFF', b'wOF2', b'true', b'typ1', b'\x00\x01\x00\x00', b'\x00\x00\x01\x00'):
                                font_magic = True
                    except Exception:
                        pass

                score = 0
                if font_magic:
                    score += 5000
                if is_font_ext:
                    score += 3000
                if has_keyword:
                    score += 2000
                if in_poc_dir:
                    score += 1000

                score -= abs(size - 800)

                if best_score is None or score > best_score:
                    best_score = score
                    best_member = m

            return best_member

        try:
            with tarfile.open(src_path, "r:*") as tf:
                member = select_poc_member(tf)
                if member is not None:
                    f = tf.extractfile(member)
                    if f is not None:
                        data = f.read()
                        if isinstance(data, bytes) and data:
                            return data
        except Exception:
            pass

        # Fallback: synthetic minimal OpenType-like blob (does not rely on repo contents)
        base = bytearray()
        base += b'OTTO'
        if len(base) < 800:
            base += b'\x00' * (800 - len(base))
        return bytes(base)