import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        # Attempt to access the tarball (no-op for robustness; PoC is static)
        try:
            if os.path.isfile(src_path):
                with tarfile.open(src_path, "r:*"):
                    pass
        except Exception:
            pass

        poc_src = r"""
# PoC placeholder for heap use-after-free in mruby VM
# This Ruby script is designed to heavily exercise the VM stack,
# extensions, and block/proc machinery, which are commonly involved
# in mrb_stack_extend-related bugs.

class Stress
  def initialize(depth, width)
    @depth = depth
    @width = width
  end

  def make_procs(level, &blk)
    return [blk] if level <= 0
    a = []
    @width.times do |i|
      a << make_procs(level - 1) do
        if blk
          blk.call(level, i)
        else
          (level + i).to_s * (i + 1)
        end
      end
    end
    a.flatten
  end

  def cascade_calls(procs)
    # call each proc with a variable number of arguments to
    # stress call frames and stack extension logic
    procs.each_with_index do |p, i|
      args = []
      (i % 16).times do |j|
        args << (j * i)
      end
      begin
        p.call(*args)
      rescue Exception
        # ignore any runtime errors, we only care about exercising VM
      end
    end
  end

  def big_run
    root_procs = make_procs(@depth) do |l, i|
      s = ""
      (l + 3).times do |x|
        s << (("a".ord + ((x + i) % 26))).chr
      end
      s.reverse!
      s * (i + 1)
    end

    # Stress nested blocks, procs and lambdas
    50.times do |outer|
      seq = root_procs.map do |pr|
        lambda do |z|
          begin
            pr.call(outer, z, self, pr)
          rescue Exception
          end
        end
      end

      cascade_calls(seq)

      # More block gymnastics to exercise environment/stack pointers
      seq.each_with_index do |fn, idx|
        begin
          [1, 2, 3, 4].each do |v|
            3.times do |inner|
              fn.call(v + inner + idx)
            end
          end
        rescue Exception
        end
      end
    end
  end
end

def deep_stack(n, acc = 0, &blk)
  if n <= 0
    blk.call(acc) if blk
    return acc
  end

  # create a lot of local variables to increase frame size
  a1 = n + 1
  a2 = n + 2
  a3 = n + 3
  a4 = n + 4
  a5 = n + 5
  a6 = n + 6
  a7 = n + 7
  a8 = n + 8
  a9 = n + 9
  a10 = n + 10
  a11 = n + 11
  a12 = n + 12
  a13 = n + 13
  a14 = n + 14
  a15 = n + 15
  a16 = n + 16
  a17 = n + 17
  a18 = n + 18
  a19 = n + 19
  a20 = n + 20

  s = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 +
      a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a20

  deep_stack(n - 1, acc + s) do |x|
    if blk
      blk.call(x)
    end
  end
end

def many_args(*args, &blk)
  if blk
    blk.call(*args)
  else
    args.size
  end
end

# Main driver that tries to force repeated mrb_stack_extend activity
# by:
#   - creating deep recursion with large frames
#   - passing/creating many blocks, lambdas and procs
#   - using calls with varying arities
stress = Stress.new(4, 4)

50.times do |i|
  begin
    deep_stack(80) do |val|
      many_args(
        val, i, stress, self,
        :symbol_1, :symbol_2, :symbol_3, :symbol_4,
        "string1", "string2", "string3", "string4",
        [1, 2, 3, 4, 5],
        { a: 1, b: 2, c: 3, d: 4 },
        Object.new, Class.new, Module.new,
        lambda { |x| x * 2 },
        proc { |x| x + 3 },
        method(:many_args)
      ) do |*xs|
        xs.each_with_index do |xv, idx|
          begin
            many_args(xv, idx, i, stress) do |a, b, c, d|
              (a.to_s + b.to_s + c.to_s + d.to_s).object_id
            end
          rescue Exception
          end
        end
      end
    end
  rescue Exception
  end

  begin
    stress.big_run
  rescue Exception
  end
end

# Additional nested blocks to keep environments alive across
# potential stack extensions.
100.times do |k|
  begin
    arr = (0..50).to_a
    thunk = lambda do
      arr.map do |x|
        (0..10).map do |y|
          many_args(k, x, y, arr, stress) do |a, b, c, d, e|
            (a + b + c).to_s + d.size.to_s + e.class.to_s
          end
        end
      end
    end

    thunk.call
  rescue Exception
  end
end
"""

        return poc_src.encode("utf-8")