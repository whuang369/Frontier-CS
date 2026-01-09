import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        target_len = 71298

        header = b"USBREDIR_POC\n" + b"\x00" * 16

        def make_block16(n_frames=256, frame_size=16, fill_byte=0x41):
            # 16-bit little-endian length-prefixed frames
            parts = []
            frame = bytes([fill_byte]) * frame_size
            prefix = struct.pack("<H", frame_size)
            one = prefix + frame
            parts.extend([one] * n_frames)
            return b"".join(parts)

        def make_block32(n_frames=128, frame_size=64, fill_byte=0x42):
            # 32-bit little-endian length-prefixed frames
            parts = []
            frame = bytes([fill_byte]) * frame_size
            prefix = struct.pack("<I", frame_size)
            one = prefix + frame
            parts.extend([one] * n_frames)
            return b"".join(parts)

        block16 = make_block16()
        block32 = make_block32()

        # Additional mixed block to present diverse structure
        mixed = []
        mixed.append(block16)
        mixed.append(b"USBREDIR" * 64)
        mixed.append(block32)
        mixed.append(b"\x55\x53\x42\x52" * 128)  # "USBR" magic repeated
        mixed_block = b"".join(mixed)

        # Build up until we exceed target length
        data = bytearray()
        data.extend(header)
        while len(data) < target_len:
            data.extend(mixed_block)
            if len(data) >= target_len:
                break
            data.extend(block16)
            if len(data) >= target_len:
                break
            data.extend(block32)

        return bytes(data[:target_len])