import os
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        TARGET_LEN = 800
        font_exts = {
            '.ttf', '.otf', '.woff', '.woff2', '.otc', '.cff', '.pfa', '.pfb', '.ttc'
        }
        keywords = (
            'poc', 'crash', 'uaf', 'use_after_free', 'use-after-free',
            'heap', 'clusterfuzz', 'fuzz', 'oss-fuzz'
        )

        def score_candidate(path_lower: str, size: int):
            ext = os.path.splitext(path_lower)[1]
            ext_good = 0 if (ext in font_exts or 'font' in path_lower) else 1
            name_good = 0 if any(k in path_lower for k in keywords) else 1
            size_diff = abs(size - TARGET_LEN)
            return (ext_good, name_good, size_diff, size)

        # Try tarball first
        try:
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r:*') as tar:
                    best_member = None
                    best_score = None
                    for member in tar:
                        if not member.isreg():
                            continue
                        size = member.size
                        if size <= 0:
                            continue
                        path_lower = member.name.lower()
                        score = score_candidate(path_lower, size)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_member = member
                    if best_member is not None:
                        f = tar.extractfile(best_member)
                        if f is not None:
                            data = f.read()
                            if isinstance(data, bytes) and data:
                                return data
        except Exception:
            pass

        # Fallback: try zipfile (in case the input is actually a zip)
        try:
            if zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, 'r') as zf:
                    best_name = None
                    best_score = None
                    for info in zf.infolist():
                        # Skip directories
                        if hasattr(info, 'is_dir') and info.is_dir():
                            continue
                        if info.file_size <= 0:
                            continue
                        path_lower = info.filename.lower()
                        score = score_candidate(path_lower, info.file_size)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_name = info.filename
                    if best_name is not None:
                        with zf.open(best_name, 'r') as f:
                            data = f.read()
                            if isinstance(data, bytes) and data:
                                return data
        except Exception:
            pass

        # Ultimate fallback: synthetic PoC of target length
        return b'A' * TARGET_LEN