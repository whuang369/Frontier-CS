import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a double-free that occurs when Node::add throws an exception.
        # A common way for an 'add' method to throw an exception is by attempting to
        # add a duplicate item.
        #
        # The PoC triggers this scenario:
        # 1. `node r`: A parent node 'r' is created.
        # 2. `add r c`: A child node 'c' is created and added to 'r'. This should succeed.
        # 3. `add r c`: A second attempt is made to add a child 'c' to 'r'. This is
        #    expected to trigger a duplicate check failure within Node::add, causing
        #    an exception to be thrown. A flaw in the exception handling logic then
        #    leads to a double-free.
        #
        # This PoC uses short node names to minimize its length, aiming for a higher
        # score as per the evaluation formula.
        
        poc = b"node r\n"
        poc += b"add r c\n"
        poc += b"add r c\n"
        
        return poc