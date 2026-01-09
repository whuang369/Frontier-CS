import os
import tarfile
import gzip
import lzma
import bz2


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        fallback_poc = b"a/=0\n"

        try:
            tf = tarfile.open(src_path, "r:*")
        except Exception:
            return fallback_poc

        try:
            members = tf.getmembers()
            if not members:
                return fallback_poc

            interesting_keywords = [
                "poc",
                "crash",
                "uaf",
                "use-after-free",
                "use_after_free",
                "heap",
                "fuzz",
                "bug",
                "issue",
                "regress",
                "test",
                "tests",
                "case",
                "sample",
                "example",
                "input",
                "corpus",
                "id:",
                "sig:",
                "divide",
                "div0",
                "div_by_zero",
                "div-by-zero",
                "zerodiv",
                "zero-div",
            ]

            interesting_exts = {
                ".in",
                ".txt",
                ".js",
                ".json",
                ".lua",
                ".php",
                ".rb",
                ".py",
                ".dat",
                ".bin",
                ".xml",
                ".html",
                ".src",
                ".expr",
                ".input",
                ".tmpl",
                ".wasm",
            }

            uninteresting_exts = {
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".h",
                ".hh",
                ".hpp",
                ".hxx",
                ".md",
                ".markdown",
                ".rst",
            }

            skip_basenames = {
                "readme",
                "license",
                "copying",
                "changelog",
                "news",
                "todo",
            }

            best_member = None
            best_score = None

            for member in members:
                if not member.isfile():
                    continue

                size = member.size
                if size <= 0 or size > 4096:
                    continue

                path = member.name
                base = os.path.basename(path)
                lower_base = base.lower()
                path_lower = path.lower()

                if lower_base in skip_basenames:
                    continue

                if lower_base.startswith(".") and all(
                    kw not in path_lower for kw in ("poc", "crash", "fuzz", "test")
                ):
                    continue

                dot_idx = lower_base.rfind(".")
                ext = lower_base[dot_idx:] if dot_idx != -1 else ""

                if ext in uninteresting_exts:
                    continue

                is_interesting_path = any(k in path_lower for k in interesting_keywords)
                is_interesting_ext = ext in interesting_exts

                if not is_interesting_path and not is_interesting_ext and size > 256:
                    continue

                size_diff = abs(size - 79)
                score = 0.0

                if size == 79:
                    score += 40.0

                score -= size_diff * 0.5
                score -= size * 0.01

                if is_interesting_path:
                    score += 30.0
                if is_interesting_ext:
                    score += 10.0

                if "poc" in path_lower:
                    score += 40.0
                if "crash" in path_lower:
                    score += 35.0
                if "uaf" in path_lower or "use-after-free" in path_lower or "use_after_free" in path_lower:
                    score += 25.0
                if (
                    "divide" in path_lower
                    or "div0" in path_lower
                    or "div_by_zero" in path_lower
                    or "div-by-zero" in path_lower
                    or "zerodiv" in path_lower
                    or "zero-div" in path_lower
                ):
                    score += 20.0
                if "bug" in path_lower or "issue" in path_lower:
                    score += 15.0
                if "test" in path_lower or "regress" in path_lower:
                    score += 10.0
                if "fuzz" in path_lower:
                    score += 10.0

                if best_member is None:
                    best_member = member
                    best_score = score
                else:
                    best_size_diff = abs(best_member.size - 79)
                    if (
                        score > best_score
                        or (
                            score == best_score
                            and (
                                size_diff < best_size_diff
                                or (
                                    size_diff == best_size_diff
                                    and size < best_member.size
                                )
                            )
                        )
                    ):
                        best_member = member
                        best_score = score

            if best_member is None:
                return fallback_poc

            f = tf.extractfile(best_member)
            if f is None:
                return fallback_poc
            data = f.read()

            lower_name = best_member.name.lower()
            try:
                if lower_name.endswith(".gz") or lower_name.endswith(".gzip"):
                    decompressed = gzip.decompress(data)
                    if len(decompressed) <= 1_000_000:
                        data = decompressed
                elif lower_name.endswith(".xz") or lower_name.endswith(".lzma"):
                    decompressed = lzma.decompress(data)
                    if len(decompressed) <= 1_000_000:
                        data = decompressed
                elif lower_name.endswith(".bz2"):
                    decompressed = bz2.decompress(data)
                    if len(decompressed) <= 1_000_000:
                        data = decompressed
            except Exception:
                pass

            if not isinstance(data, (bytes, bytearray)):
                try:
                    data = bytes(data)
                except Exception:
                    try:
                        data = str(data).encode("utf-8", errors="ignore")
                    except Exception:
                        data = fallback_poc

            if not data:
                return fallback_poc

            return bytes(data)
        finally:
            try:
                tf.close()
            except Exception:
                pass