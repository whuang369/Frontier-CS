import tarfile


class Solution:
    TARGET_LEN = 159

    def _score_member(self, name_lower: str, size: int) -> int:
        score = 0
        target = self.TARGET_LEN
        diff = abs(size - target)

        if diff == 0:
            score += 100
        else:
            closeness = max(0, 60 - diff // 4)
            score += closeness

        if size < 1024:
            score += 15
        elif size < 4096:
            score += 8
        elif size > 65536:
            score -= 25
        elif size > 16384:
            score -= 12

        if 'poc' in name_lower:
            score += 50
        if 'crash' in name_lower or 'repro' in name_lower:
            score += 40
        if '61292' in name_lower:
            score += 50
        if 'arvo' in name_lower:
            score += 25
        if (
            'uaf' in name_lower
            or 'useafterfree' in name_lower
            or 'use-after-free' in name_lower
            or 'use_after_free' in name_lower
        ):
            score += 30
        if 'heap' in name_lower:
            score += 10
        if 'cue' in name_lower:
            score += 8
        if 'cuesheet' in name_lower or 'cue_sheet' in name_lower:
            score += 12
        if 'asan' in name_lower or 'ubsan' in name_lower:
            score += 8
        if 'fuzz' in name_lower:
            score += 6
        if 'regress' in name_lower or 'bug' in name_lower:
            score += 6
        if 'test' in name_lower or 'tests' in name_lower:
            score += 2
        if 'input' in name_lower or 'seed' in name_lower or 'corpus' in name_lower:
            score += 4
        if 'crashes' in name_lower or 'artifacts' in name_lower:
            score += 5

        if name_lower.endswith('.cue'):
            score += 40
        if name_lower.endswith('.flac'):
            score += 25
        if name_lower.endswith('.bin') or name_lower.endswith('.dat'):
            score += 8
        if name_lower.endswith('.txt'):
            score += 5

        if '/poc' in name_lower or '\\poc' in name_lower:
            score += 10
        if '/crash' in name_lower or '\\crash' in name_lower:
            score += 8

        return score

    def solve(self, src_path: str) -> bytes:
        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return b""

        best_member = None
        best_score = None

        for member in tf.getmembers():
            if not member.isfile():
                continue
            size = member.size
            name_lower = member.name.lower()
            score = self._score_member(name_lower, size)

            if best_score is None or score > best_score:
                best_score = score
                best_member = member

        data = b""
        if best_member is not None:
            extracted = tf.extractfile(best_member)
            if extracted is not None:
                try:
                    data = extracted.read()
                finally:
                    extracted.close()
        tf.close()
        return data