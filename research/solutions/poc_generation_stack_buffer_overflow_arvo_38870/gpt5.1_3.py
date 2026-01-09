import io
import zipfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
            filename = "A" * 300 + ".obj"
            zf.writestr(filename, b"content")
        return buffer.getvalue()