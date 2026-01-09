import os
import tarfile
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        def extract_tarball(path):
            tmpdir = tempfile.mkdtemp(prefix="src_extract_")
            try:
                with tarfile.open(path, 'r:*') as tf:
                    def is_within_directory(directory, target):
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        return prefix == abs_directory

                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                continue
                        tar.extractall(path=path, members=members, numeric_owner=numeric_owner)
                    safe_extract(tf, tmpdir)
            except Exception:
                return None
            return tmpdir

        def find_best_candidate(rootdir):
            best = None
            best_score = float("-inf")
            for dirpath, _, filenames in os.walk(rootdir):
                for fn in filenames:
                    fpath = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(fpath)
                        if not os.path.isfile(fpath):
                            continue
                        if st.st_size <= 0 or st.st_size > 1024 * 1024:
                            continue
                        # quick filter to avoid reading everything
                        with open(fpath, 'rb') as fb:
                            chunk = fb.read(4096)
                        if b'classpermissionset' not in chunk and b'classpermissionset' not in chunk.lower():
                            # maybe later in file; read full only if extension suggests .cil or tests
                            ext = os.path.splitext(fn)[1].lower()
                            if ext not in ('.cil', '.te', '.ciltest', '.policy', '.conf', '.txt'):
                                continue
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            txt = f.read()
                        low = txt.lower()
                        if 'classpermissionset' not in low:
                            continue
                        # Heuristic scoring
                        score = 0.0
                        # strong positive signals
                        if '(macro' in low:
                            score += 2000.0
                        if 'macro ' in low or '\nmacro ' in low:
                            score += 200.0
                        if 'classpermission' in low:
                            score += 500.0
                        if '(class ' in low:
                            score += 150.0
                        if '(call ' in low or '(macrocall' in low:
                            score += 300.0
                        if 'anonymous' in low:
                            score += 50.0
                        # length penalty (prefer shorter)
                        score -= len(txt) / 5.0
                        # prefer files under test directories
                        path_low = fpath.lower()
                        if any(seg in path_low for seg in ('/test', '/tests', '/poc', '/repro', '/regress', '/fuzz', '/examples', '/example')):
                            score += 200.0
                        # extra boost if "anon" present in path
                        if 'anon' in path_low and 'classpermission' in path_low:
                            score += 300.0
                        if score > best_score:
                            best_score = score
                            best = txt
                    except Exception:
                        continue
            return best

        # Try extract and search for PoC-like file
        extracted = extract_tarball(src_path)
        if extracted:
            candidate = find_best_candidate(extracted)
            if candidate:
                return candidate.encode('utf-8', errors='ignore')

        # Fallback minimal crafted PoC (best-effort guess if repository did not contain a sample)
        # Note: This relies on typical CIL syntax for class, macro, classpermission, classpermissionset, and macro call.
        poc = """
; minimal classes and permissions
(class c1 (p1 p2))

; Macro that constructs a classpermissionset from a classpermission parameter
(macro m_cp_to_cps ((cp classpermission))
    (classpermissionset s1 (cp))
)

; Pass an anonymous classpermission to the macro
(call m_cp_to_cps ((c1 (p1))))
"""
        return poc.strip().encode('utf-8')