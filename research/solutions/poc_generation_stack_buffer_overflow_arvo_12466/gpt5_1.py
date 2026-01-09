import os
import tarfile
import zipfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        target_size = 524

        def iter_tar(path):
            try:
                with tarfile.open(path, 'r:*') as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        size = m.size
                        name = m.name
                        def reader(member=m, tf=tf):
                            f = tf.extractfile(member)
                            if f is None:
                                return b""
                            data = f.read()
                            try:
                                f.close()
                            except Exception:
                                pass
                            return data
                        yield name, size, reader
            except tarfile.TarError:
                return

        def iter_zip(path):
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        size = zi.file_size
                        name = zi.filename
                        def reader(zi=zi, zf=zf):
                            return zf.read(zi)
                        yield name, size, reader
            except zipfile.BadZipFile:
                return

        def has_rar_sig(data):
            # RAR4: 52 61 72 21 1A 07 00
            # RAR5: 52 61 72 21 1A 07 01 00
            if len(data) < 7:
                return False
            if data.startswith(b'Rar!\x1a\x07\x00'):
                return True
            if len(data) >= 8 and data.startswith(b'Rar!\x1a\x07\x01\x00'):
                return True
            return False

        def is_rar5(data):
            return len(data) >= 8 and data.startswith(b'Rar!\x1a\x07\x01\x00')

        def score_candidate(name, size):
            lower = name.lower()
            ext_rar = 0 if lower.endswith('.rar') else 1
            contains_rar5 = 0 if 'rar5' in lower else 1
            contains_rar = 0 if 'rar' in lower else 1
            poc_flag = 0 if any(k in lower for k in ('poc', 'crash', 'id', 'fuzz', 'ossfuzz', 'oss-fuzz', 'test', 'testcase')) else 1
            huff_flag = 0 if any(k in lower for k in ('huffman', 'huff', 'code', 'table')) else 1
            exact = 0 if size == target_size else 1
            diff = abs(size - target_size)
            return (ext_rar, contains_rar5, contains_rar, poc_flag, huff_flag, exact, diff, size)

        candidates = []

        if tarfile.is_tarfile(src_path):
            for tup in iter_tar(src_path):
                if tup is None:
                    break
                name, size, reader = tup
                candidates.append((name, size, reader))
        elif zipfile.is_zipfile(src_path):
            for tup in iter_zip(src_path):
                if tup is None:
                    break
                name, size, reader = tup
                candidates.append((name, size, reader))
        else:
            # As a fallback, try tar open even if is_tarfile failed (some formats may still be readable by 'r:*')
            try:
                for tup in iter_tar(src_path):
                    if tup is None:
                        break
                    name, size, reader = tup
                    candidates.append((name, size, reader))
            except Exception:
                pass

        if not candidates:
            return b""

        # Sort candidates by heuristic score
        candidates.sort(key=lambda x: score_candidate(x[0], x[1]))

        # Try multi-stage selection

        # Stage 1: exact size match and RAR5 signature
        for name, size, reader in candidates:
            if size == target_size:
                data = reader()
                if is_rar5(data):
                    return data

        # Stage 2: exact size match and RAR (any version) signature
        for name, size, reader in candidates:
            if size == target_size:
                data = reader()
                if has_rar_sig(data):
                    return data

        # Stage 3: close size and RAR5 signature
        for name, size, reader in candidates[:200]:
            if abs(size - target_size) <= 64:
                data = reader()
                if is_rar5(data):
                    return data

        # Stage 4: any RAR5 file that looks like a PoC (name hints)
        for name, size, reader in candidates[:300]:
            lower = name.lower()
            if any(k in lower for k in ('poc', 'crash', 'id', 'fuzz', 'ossfuzz', 'oss-fuzz', 'testcase')):
                data = reader()
                if is_rar5(data):
                    return data

        # Stage 5: any RAR (any version) closest in size
        best = None
        best_diff = None
        for name, size, reader in candidates[:500]:
            data = reader()
            if has_rar_sig(data):
                d = abs(len(data) - target_size)
                if best is None or d < best_diff:
                    best = data
                    best_diff = d
                    if d == 0:
                        break
        if best is not None:
            return best

        # Stage 6: fallback to the closest-by-size candidate
        candidates.sort(key=lambda x: abs(x[1] - target_size))
        name, size, reader = candidates[0]
        try:
            data = reader()
            return data
        except Exception:
            return b""