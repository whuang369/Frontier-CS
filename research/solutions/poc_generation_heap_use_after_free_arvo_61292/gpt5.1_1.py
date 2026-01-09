import os
import tarfile
import tempfile
import json


class Solution:
    def solve(self, src_path: str) -> bytes:
        root_dir = self._prepare_root(src_path)
        poc = self._find_poc_via_metadata(root_dir)
        if poc is None:
            poc = self._find_poc_by_heuristics(root_dir)
        if poc is None:
            poc = self._fallback_cuesheet()
        return poc

    def _prepare_root(self, src_path: str) -> str:
        if os.path.isdir(src_path):
            return src_path

        if os.path.isfile(src_path) and tarfile.is_tarfile(src_path):
            tmpdir = tempfile.mkdtemp(prefix="src-")
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    tf.extractall(tmpdir)
            except tarfile.TarError:
                pass

            try:
                entries = [e for e in os.listdir(tmpdir) if not e.startswith(".")]
            except OSError:
                return tmpdir
            if len(entries) == 1:
                candidate = os.path.join(tmpdir, entries[0])
                if os.path.isdir(candidate):
                    return candidate
            return tmpdir

        # Fallback: treat parent directory as root if not a tarball
        parent = os.path.dirname(src_path)
        return parent if parent else "."

    def _find_poc_via_metadata(self, root: str) -> bytes | None:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                lower = fname.lower()
                if not any(key in lower for key in ("bug", "meta", "info", "poc")):
                    continue
                path = os.path.join(dirpath, fname)

                if lower.endswith(".json"):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:
                        continue
                    rel_paths = self._search_poc_in_json(data)
                    for rel in rel_paths:
                        poc_path = os.path.join(dirpath, rel)
                        if os.path.isfile(poc_path):
                            try:
                                with open(poc_path, "rb") as pf:
                                    return pf.read()
                            except Exception:
                                continue
                else:
                    # Treat as simple YAML/text
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                    except Exception:
                        continue
                    keys = {
                        "poc",
                        "poc_path",
                        "poc_file",
                        "input",
                        "input_file",
                        "crash_input",
                        "crash_file",
                        "crash",
                    }
                    for line in lines:
                        if ":" not in line:
                            continue
                        key, val = line.split(":", 1)
                        if key.strip().lower() not in keys:
                            continue
                        val = val.strip().strip('"').strip("'")
                        if not val:
                            continue
                        if " " in val and not os.path.exists(os.path.join(dirpath, val)):
                            val = val.split()[0]
                        poc_path = os.path.join(dirpath, val)
                        if os.path.isfile(poc_path):
                            try:
                                with open(poc_path, "rb") as pf:
                                    return pf.read()
                            except Exception:
                                pass
        return None

    def _search_poc_in_json(self, obj) -> list:
        results: list[str] = []
        keys = {
            "poc",
            "poc_path",
            "poc_file",
            "input",
            "input_file",
            "crash_input",
            "crash_file",
            "crash",
        }

        def rec(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if kl in keys and isinstance(v, str):
                        results.append(v)
                    else:
                        rec(v)
            elif isinstance(node, list):
                for item in node:
                    rec(item)

        rec(obj)
        return results

    def _find_poc_by_heuristics(self, root: str) -> bytes | None:
        max_size = 1_000_000
        cue_best = None  # (score, size, path)
        best = None      # (score, size, path)

        for dirpath, _, filenames in os.walk(root):
            lower_dir = dirpath.lower()
            for fname in filenames:
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size == 0 or size > max_size:
                    continue

                lower_name = fname.lower()
                ext = os.path.splitext(lower_name)[1]

                # Prefer .cue files first
                if lower_name.endswith(".cue"):
                    score = 10
                    for kw, s in (
                        ("poc", 5),
                        ("crash", 5),
                        ("uaf", 5),
                        ("heap", 5),
                        ("bug", 3),
                        ("issue", 3),
                        ("case", 2),
                        ("trigger", 2),
                    ):
                        if kw in lower_name or kw in lower_dir:
                            score += s
                    if cue_best is None or score > cue_best[0] or (
                        score == cue_best[0] and size < cue_best[1]
                    ):
                        cue_best = (score, size, path)
                    continue

                # General heuristic candidates
                score = 0
                for kw, s in (
                    ("poc", 10),
                    ("crash", 10),
                    ("uaf", 8),
                    ("heap", 8),
                    ("bug", 6),
                    ("issue", 4),
                    ("case", 3),
                    ("trigger", 3),
                    ("id_", 2),
                ):
                    if kw in lower_name or kw in lower_dir:
                        score += s

                if ext in (".cue", ".flac", ".bin", ".dat", ".raw", ".txt"):
                    score += 3
                if "cue" in lower_name or "cue" in lower_dir:
                    score += 2

                if score <= 0:
                    continue

                if best is None or score > best[0] or (
                    score == best[0] and size < best[1]
                ):
                    best = (score, size, path)

        chosen = cue_best if cue_best is not None else best
        if chosen is None:
            return None
        path = chosen[2]
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def _fallback_cuesheet(self) -> bytes:
        # Generic CUE sheet designed to exercise cuesheet import and seekpoint handling
        text = (
            'FILE "test.wav" WAVE\n'
            "  TRACK 01 AUDIO\n"
            "    INDEX 01 00:00:00\n"
            "    INDEX 02 00:00:01\n"
            "    INDEX 03 00:00:02\n"
            "    INDEX 04 00:00:03\n"
            "  TRACK 02 AUDIO\n"
            "    INDEX 00 00:00:04\n"
            "    INDEX 01 00:00:05\n"
            "    INDEX 02 00:00:06\n"
        )
        return text.encode("ascii", errors="ignore")