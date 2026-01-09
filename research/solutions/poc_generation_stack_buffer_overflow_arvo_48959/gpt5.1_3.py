import gzip


class Solution:
    def solve(self, src_path: str) -> bytes:
        def extract_deflate(gz_bytes: bytes) -> bytes:
            if len(gz_bytes) < 18:
                return b""
            if gz_bytes[0] != 0x1F or gz_bytes[1] != 0x8B:
                return b""
            if gz_bytes[2] != 8:
                return b""
            flg = gz_bytes[3]
            idx = 10  # base header size

            # FEXTRA
            if flg & 4:
                if idx + 2 > len(gz_bytes):
                    return b""
                xlen = gz_bytes[idx] | (gz_bytes[idx + 1] << 8)
                idx += 2 + xlen

            # FNAME
            if flg & 8:
                while idx < len(gz_bytes) and gz_bytes[idx] != 0:
                    idx += 1
                idx += 1

            # FCOMMENT
            if flg & 16:
                while idx < len(gz_bytes) and gz_bytes[idx] != 0:
                    idx += 1
                idx += 1

            # FHCRC
            if flg & 2:
                idx += 2

            if idx >= len(gz_bytes) - 8:
                return b""
            return gz_bytes[idx:-8]

        best_gz = None
        best_len = None
        target_len = 27

        # Primary search: small 'A'*n payloads, various compression levels
        max_n_small = 1024
        for level in (9, 6, 1):
            for n in range(1, max_n_small + 1):
                data = b"A" * n
                gz = gzip.compress(data, compresslevel=level)
                defl = extract_deflate(gz)
                if not defl:
                    continue
                btype = (defl[0] >> 1) & 0x3
                if btype == 2:  # dynamic Huffman
                    L = len(gz)
                    if best_len is None or L < best_len:
                        best_len = L
                        best_gz = gz
                        if best_len <= target_len:
                            return best_gz

        if best_gz is None or best_len is None:
            best_len = 1 << 60

        # Extended search if we haven't found a small enough dynamic-block gzip
        if best_len > 60:
            # Try longer runs of 'A' at high compression level
            for n in range(max_n_small + 1, 8193):
                data = b"A" * n
                gz = gzip.compress(data, compresslevel=9)
                defl = extract_deflate(gz)
                if not defl:
                    continue
                btype = (defl[0] >> 1) & 0x3
                if btype == 2:
                    L = len(gz)
                    if L < best_len:
                        best_len = L
                        best_gz = gz
                        if best_len <= target_len:
                            return best_gz
                if n % 1024 == 0 and best_gz is not None and best_len <= 80:
                    break

            # If still nothing great, try a few other simple patterns
            if best_gz is None:
                patterns = [b"AB", b"ABC", b"Hello, world! "]
                for pattern in patterns:
                    for reps in range(2, 512):
                        data = pattern * reps
                        gz = gzip.compress(data, compresslevel=9)
                        defl = extract_deflate(gz)
                        if not defl:
                            continue
                        btype = (defl[0] >> 1) & 0x3
                        if btype == 2:
                            L = len(gz)
                            if L < best_len:
                                best_len = L
                                best_gz = gz
                                if best_len <= target_len:
                                    return best_gz

        if best_gz is None:
            best_gz = gzip.compress(b"A" * 100, compresslevel=9)
        return best_gz