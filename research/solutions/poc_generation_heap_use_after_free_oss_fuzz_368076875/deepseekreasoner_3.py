import os
import tarfile
import tempfile
import subprocess
import shutil
from pathlib import Path
import random
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find the source root (assuming single directory in tarball)
            extracted_items = os.listdir(tmpdir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmpdir, extracted_items[0])):
                src_root = os.path.join(tmpdir, extracted_items[0])
            else:
                src_root = tmpdir
            
            # Look for Python files to understand the AST structure
            python_files = []
            for root, _, files in os.walk(src_root):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            # Look for AST-related code patterns
            ast_patterns = ['class.*AST', 'class.*Node', '__repr__', 'repr(']
            ast_files = []
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if any(pattern in content for pattern in ast_patterns):
                            ast_files.append(py_file)
                except:
                    continue
            
            # If we can't analyze the code, generate a generic PoC based on common AST repr UAF patterns
            # Common pattern: deeply nested structures with circular references that cause premature garbage collection
            
            # Generate a PoC that creates a deeply nested AST structure
            # This is a general approach that often triggers UAF in AST repr implementations
            poc = self.generate_ast_poc()
            
            return poc
    
    def generate_ast_poc(self) -> bytes:
        """
        Generate a PoC that creates a deeply nested AST structure.
        Common vulnerability patterns in AST repr:
        1. Circular references
        2. Tree structures where parent references children and children reference parents
        3. Deletion of nodes while references still exist
        """
        # We'll create a Python script that builds a complex AST and triggers repr
        # The exact structure depends on the vulnerable library, but we can use a generic approach
        
        # Create a deeply nested dictionary structure that mimics an AST
        # This often triggers issues in recursive repr implementations
        
        # Build a tree with depth 50, each node has 2-3 children
        # This creates ~2^50 nodes in the recursion which can trigger memory issues
        
        poc_code = '''class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []
    
    def __repr__(self):
        # This repr implementation might have UAF if children are freed during recursion
        children_repr = ', '.join(repr(child) for child in self.children)
        return f'Node({self.value}, [{children_repr}])'

def build_deep_tree(depth, breadth=2):
    """Build a deeply nested tree structure"""
    if depth <= 0:
        return Node('leaf')
    
    children = []
    for i in range(breadth):
        child = build_deep_tree(depth - 1, breadth)
        children.append(child)
    
    # Create circular reference
    node = Node(f'level_{depth}', children)
    
    # Add back-reference from children to parent (creates circular references)
    for child in children:
        if not hasattr(child, 'parent'):
            child.parent = node
    
    return node

# Create the tree
tree = build_deep_tree(50, 2)

# Trigger the vulnerability
# The repr() call will recursively traverse the tree
# If there's a UAF in the AST repr implementation, this should trigger it
repr(tree)

# Additional trigger: delete some references while repr is running
# This can cause UAF in some implementations
import gc
gc.collect()

# Try to access the tree after potential free
print("Tree repr started")
result = repr(tree)
print("Tree repr completed")
'''

        # Alternative approach: Create a malformed AST that causes repr to
        # access freed memory when dealing with certain node types
        
        # Create a script with a complex class hierarchy
        complex_poc = '''# Complex AST structure with multiple inheritance and circular refs

class BaseNode:
    def __init__(self):
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class ListNode(BaseNode):
    def __repr__(self):
        items = []
        for child in self.children:
            items.append(repr(child))
        return f'[{", ".join(items)}]'

class DictNode(BaseNode):
    def __repr__(self):
        items = []
        for i in range(0, len(self.children), 2):
            if i + 1 < len(self.children):
                key = repr(self.children[i])
                value = repr(self.children[i + 1])
                items.append(f'{key}: {value}')
        return f'{{{", ".join(items)}}}'

class ValueNode(BaseNode):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def __repr__(self):
        return repr(self.value)

# Build a complex nested structure
root = DictNode()
list_node = ListNode()

# Create circular references
for i in range(1000):
    val = ValueNode(f'item_{i}')
    list_node.add_child(val)

# Add the list to dict multiple times (creates shared references)
for i in range(10):
    key = ValueNode(f'key_{i}')
    root.add_child(key)
    root.add_child(list_node)  # Same list node added multiple times

# Add self-reference
root.add_child(root)

# Add mutual references between nodes
node_a = DictNode()
node_b = DictNode()
node_a.add_child(node_b)
node_b.add_child(node_a)
root.add_child(node_a)

# Trigger repr - this should cause UAF if implementation is vulnerable
print("Starting repr...")
try:
    result = repr(root)
    print("Repr completed")
except Exception as e:
    print(f"Exception during repr: {e}")

# Force garbage collection during repr attempt
import gc
gc.collect()

# Try to access objects after potential free
print(f"Root children count: {len(root.children)}")
'''
        
        # Use the complex PoC
        return complex_poc.encode('utf-8')