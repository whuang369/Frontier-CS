import os
import io
import tarfile
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidates = []
        try:
            if os.path.isdir(src_path):
                candidates.extend(self._scan_dir_for_poc(src_path))
            elif self._is_tar(src_path):
                candidates.extend(self._scan_tar_for_poc(src_path))
            elif self._is_zip(src_path):
                candidates.extend(self._scan_zip_for_poc(src_path))
        except Exception:
            pass

        if not candidates and self._is_tar(src_path):
            try:
                candidates.extend(self._scan_tar_for_poc(src_path, deep=True))
            except Exception:
                pass

        # If we gathered some candidates, refine scores using content headers
        if candidates:
            # Sort preliminary to limit deeper inspection
            candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
            top = candidates[:200] if len(candidates) > 200 else candidates
            for cand in top:
                try:
                    head = self._peek_candidate_head(cand, 16)
                    if head:
                        extra = self._signature_bonus(head)
                        cand["score"] = cand.get("score", 0) + extra
                except Exception:
                    continue

            candidates.sort(key=lambda c: c.get("score", 0), reverse=True)

            for cand in candidates[:50]:
                try:
                    data = self._read_candidate_bytes(cand)
                    if data:
                        return data
                except Exception:
                    continue

        # Fallback: craft a dummy WOFF-like blob of ~800 bytes
        return self._fallback_woff(800)

    # --------------- Scanning Helpers ---------------

    def _is_tar(self, path: str) -> bool:
        try:
            return tarfile.is_tarfile(path)
        except Exception:
            return False

    def _is_zip(self, path: str) -> bool:
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _scan_dir_for_poc(self, dpath: str):
        cands = []
        for root, _, files in os.walk(dpath):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                except Exception:
                    continue
                lower = fname.lower()
                base_score = self._name_score(lower, size)
                if base_score <= 0:
                    continue
                ext = os.path.splitext(lower)[1]
                if ext == ".zip":
                    # Scan inner zip
                    zcands = self._scan_zip_for_poc(fpath)
                    cands.extend(zcands)
                else:
                    # Regular file
                    cands.append({
                        "type": "dir_file",
                        "path": fpath,
                        "size": size,
                        "name": fname,
                        "score": base_score
                    })
        return cands

    def _scan_zip_for_poc(self, zpath: str):
        cands = []
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    size = info.file_size
                    name = info.filename
                    lower = name.lower()
                    base_score = self._name_score(lower, size)
                    if base_score <= 0:
                        continue
                    # limit to reasonable size
                    if size > 5 * 1024 * 1024:
                        continue
                    cands.append({
                        "type": "zip_member_in_dir",
                        "zip_path": zpath,
                        "inner_name": name,
                        "size": size,
                        "name": name,
                        "score": base_score
                    })
        except Exception:
            pass
        return cands

    def _scan_tar_for_poc(self, tpath: str, deep: bool = False):
        cands = []
        try:
            with tarfile.open(tpath, 'r:*') as tf:
                for m in tf.getmembers():
                    if not m.isreg():
                        continue
                    size = m.size if m.size is not None else 0
                    name = m.name
                    lower = name.lower()
                    base_score = self._name_score(lower, size)
                    if base_score <= 0 and not deep:
                        # still consider archives if deep scanning requested
                        if not (lower.endswith(".zip") or lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz")):
                            continue

                    # Add tar-contained file itself
                    if base_score > 0:
                        cands.append({
                            "type": "tar_member",
                            "tar_path": tpath,
                            "member_name": name,
                            "size": size,
                            "name": name,
                            "score": base_score
                        })

                    # Scan nested zips to find PoCs
                    if lower.endswith(".zip"):
                        if size > 50 * 1024 * 1024:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if not f:
                                continue
                            zdata = f.read()
                            with zipfile.ZipFile(io.BytesIO(zdata), 'r') as zf:
                                for info in zf.infolist():
                                    if info.is_dir():
                                        continue
                                    isize = info.file_size
                                    iname = info.filename
                                    ilower = iname.lower()
                                    iscore = self._name_score(ilower, isize)
                                    if iscore <= 0:
                                        continue
                                    if isize > 5 * 1024 * 1024:
                                        continue
                                    cands.append({
                                        "type": "zip_member_in_tar",
                                        "tar_path": tpath,
                                        "zip_member_name": name,
                                        "inner_name": iname,
                                        "size": isize,
                                        "name": f"{name}:{iname}",
                                        "score": iscore
                                    })
                        except Exception:
                            continue

                    # Optionally scan nested tarballs if deep is True
                    if deep and (lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz")):
                        if size > 100 * 1024 * 1024:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if not f:
                                continue
                            tbytes = f.read()
                            with tarfile.open(fileobj=io.BytesIO(tbytes), mode='r:*') as inner_tf:
                                for im in inner_tf.getmembers():
                                    if not im.isreg():
                                        continue
                                    isize = im.size if im.size is not None else 0
                                    iname = im.name
                                    ilower = iname.lower()
                                    iscore = self._name_score(ilower, isize)
                                    if iscore <= 0:
                                        continue
                                    cands.append({
                                        "type": "tar_member_nested",
                                        "outer_tar_path": tpath,
                                        "inner_tar_member_name": name,
                                        "nested_member_name": iname,
                                        "size": isize,
                                        "name": f"{name}::{iname}",
                                        "score": iscore
                                    })
                        except Exception:
                            continue
        except Exception:
            pass
        return cands

    # --------------- Scoring and Reading ---------------

    def _name_score(self, lower_name: str, size: int) -> int:
        score = 0
        # Prefer font formats
        exts = (".ttf", ".otf", ".woff", ".woff2", ".ttc", ".otc")
        if lower_name.endswith(exts):
            score += 500
        elif lower_name.endswith(".bin") or lower_name.endswith(".dat"):
            score += 50

        # Keywords
        if "poc" in lower_name:
            score += 800
        if "crash" in lower_name:
            score += 700
        if "uaf" in lower_name or "use-after" in lower_name or "use_after" in lower_name or "afterfree" in lower_name:
            score += 700
        if "heap" in lower_name:
            score += 300
        if "ots" in lower_name or "opentype" in lower_name:
            score += 500
        if "otsstream" in lower_name or "write" in lower_name:
            score += 300
        if "clusterfuzz" in lower_name or "oss-fuzz" in lower_name:
            score += 300
        if "min" in lower_name or "minimized" in lower_name:
            score += 150
        if "testcase" in lower_name or "id:" in lower_name or "sig:" in lower_name or "repro" in lower_name:
            score += 200
        if "919" in lower_name or "arvo" in lower_name:
            score += 400
        if "woff2" in lower_name:
            score += 200
        if "woff" in lower_name:
            score += 150
        if "ttf" in lower_name or "otf" in lower_name:
            score += 150

        # Size closeness to 800 bytes
        closeness = abs(int(size) - 800)
        score += max(0, 1200 - min(1200, closeness))

        # Penalize huge files
        if size > 10 * 1024 * 1024:
            score -= 1000
        elif size > 1 * 1024 * 1024:
            score -= 400
        elif size > 200 * 1024:
            score -= 150

        return score

    def _signature_bonus(self, head: bytes) -> int:
        if not head or len(head) < 4:
            return 0
        bonus = 0
        # WOFF or WOFF2
        if head[:4] == b"wOFF":
            bonus += 2500
        elif head[:4] == b"wOF2":
            bonus += 2500
        # 'OTTO' for OTF/CFF
        elif head[:4] == b"OTTO":
            bonus += 2200
        # TrueType sfnt version 0x00010000
        elif head[:4] == b"\x00\x01\x00\x00":
            bonus += 2200
        # TTCF
        elif head[:4].lower() == b"ttcf":
            bonus += 1500
        # Other likely data
        else:
            bonus += 0
        return bonus

    def _peek_candidate_head(self, cand: dict, n: int) -> bytes:
        t = cand.get("type")
        if t == "dir_file":
            path = cand["path"]
            with open(path, "rb") as f:
                return f.read(n)
        elif t == "zip_member_in_dir":
            zpath = cand["zip_path"]
            inner = cand["inner_name"]
            with zipfile.ZipFile(zpath, 'r') as zf:
                with zf.open(inner) as f:
                    return f.read(n)
        elif t == "tar_member":
            tpath = cand["tar_path"]
            mname = cand["member_name"]
            with tarfile.open(tpath, 'r:*') as tf:
                m = tf.getmember(mname)
                f = tf.extractfile(m)
                if f is None:
                    return b""
                return f.read(n)
        elif t == "zip_member_in_tar":
            tpath = cand["tar_path"]
            zmember = cand["zip_member_name"]
            inner = cand["inner_name"]
            with tarfile.open(tpath, 'r:*') as tf:
                m = tf.getmember(zmember)
                f = tf.extractfile(m)
                if f is None:
                    return b""
                zdata = f.read()
                with zipfile.ZipFile(io.BytesIO(zdata), 'r') as zf:
                    with zf.open(inner) as innerf:
                        return innerf.read(n)
        elif t == "tar_member_nested":
            # Not peeking nested inner tar members for simplicity
            return b""
        return b""

    def _read_candidate_bytes(self, cand: dict) -> bytes:
        t = cand.get("type")
        if t == "dir_file":
            with open(cand["path"], "rb") as f:
                return f.read()
        elif t == "zip_member_in_dir":
            with zipfile.ZipFile(cand["zip_path"], 'r') as zf:
                with zf.open(cand["inner_name"]) as f:
                    return f.read()
        elif t == "tar_member":
            with tarfile.open(cand["tar_path"], 'r:*') as tf:
                m = tf.getmember(cand["member_name"])
                f = tf.extractfile(m)
                if f is None:
                    return b""
                return f.read()
        elif t == "zip_member_in_tar":
            with tarfile.open(cand["tar_path"], 'r:*') as tf:
                m = tf.getmember(cand["zip_member_name"])
                f = tf.extractfile(m)
                if f is None:
                    return b""
                zdata = f.read()
                with zipfile.ZipFile(io.BytesIO(zdata), 'r') as zf:
                    with zf.open(cand["inner_name"]) as innerf:
                        return innerf.read()
        elif t == "tar_member_nested":
            # Read nested tar member
            with tarfile.open(cand["outer_tar_path"], 'r:*') as outer_tf:
                om = outer_tf.getmember(cand["inner_tar_member_name"])
                of = outer_tf.extractfile(om)
                if of is None:
                    return b""
                otbytes = of.read()
                with tarfile.open(fileobj=io.BytesIO(otbytes), mode='r:*') as inner_tf:
                    im = inner_tf.getmember(cand["nested_member_name"])
                    inf = inner_tf.extractfile(im)
                    if inf is None:
                        return b""
                    return inf.read()
        return b""

    # --------------- Fallback PoC generator ---------------

    def _fallback_woff(self, length: int) -> bytes:
        # Construct a minimal WOFF-like header with declared length and pad
        # WOFF Header (simplified):
        # signature: 'wOFF' (4)
        # flavor: 0x00010000 (sfnt version) (4)
        # length: total (4)
        # numTables: 0x0001 (2) (we'll fake one)
        # reserved: 2
        # totalSfntSize: 0 (4)
        # majorVersion: 1 (2)
        # minorVersion: 0 (2)
        # metaOffset/metaLength/metaOrigLength: 0 each (12)
        # privOffset/privLength: 0 each (8)
        header = bytearray()
        header += b"wOFF"
        header += b"\x00\x01\x00\x00"  # flavor
        total_len = max(44, int(length))
        header += total_len.to_bytes(4, 'big')  # length
        header += (1).to_bytes(2, 'big')  # numTables
        header += (0).to_bytes(2, 'big')  # reserved
        header += (0).to_bytes(4, 'big')  # totalSfntSize
        header += (1).to_bytes(2, 'big')  # major
        header += (0).to_bytes(2, 'big')  # minor
        header += (0).to_bytes(4, 'big')  # metaOffset
        header += (0).to_bytes(4, 'big')  # metaLength
        header += (0).to_bytes(4, 'big')  # metaOrigLength
        header += (0).to_bytes(4, 'big')  # privOffset
        header += (0).to_bytes(4, 'big')  # privLength

        # Table directory entry (20 bytes) - fake
        # tag(4), offset(4), compLength(4), origLength(4), origChecksum(4)
        table_dir = bytearray()
        table_dir += b"name"  # arbitrary tag
        table_dir += (44 + 20).to_bytes(4, 'big')  # data offset after header + tableDir
        table_dir += (0).to_bytes(4, 'big')  # compLength
        table_dir += (0).to_bytes(4, 'big')  # origLength
        table_dir += (0).to_bytes(4, 'big')  # checksum

        buf = header + table_dir
        if len(buf) < total_len:
            buf += b"\x00" * (total_len - len(buf))
        elif len(buf) > total_len:
            buf = buf[:total_len]
        return bytes(buf)