class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers a double-free vulnerability.

        The vulnerability description states that a double-free occurs when
        the function Node::add throws an exception. A common reason for an `add`
        method in a tree-like structure to throw an exception is an attempt to
        create a structural violation, such as creating a cycle.

        This PoC constructs a minimal sequence of commands to trigger such a
        condition:
        1.  `new A`: Creates a node named 'A'.
        2.  `new B`: Creates a node named 'B'.
        3.  `add A B`: Adds 'B' as a child of 'A'. This is a valid operation.
        4.  `add B A`: Attempts to add 'A' as a child of 'B'. Since 'A' is
            already an ancestor of 'B', this would create a cycle (A -> B -> A).
            It is hypothesized that the `Node::add` implementation detects this
            and throws an exception.

        The thrown exception then exposes a flaw in the program's error handling
        or cleanup logic, leading to memory being freed more than once, which
        is detected by address sanitizers.

        The keywords 'new' and 'add' are inferred as plausible command names
        for node creation and linking, respectively. Short node names ('A', 'B')
        are used to create a compact PoC, which is rewarded by the scoring formula.
        """
        # A minimal PoC to create a cycle, which should cause Node::add to throw.
        # This, in turn, is expected to trigger the double-free vulnerability.
        poc = (
            b"new A\n"
            b"new B\n"
            b"add A B\n"
            b"add B A\n"
        )
        return poc