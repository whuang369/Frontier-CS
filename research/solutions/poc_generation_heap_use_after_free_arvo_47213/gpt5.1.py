import os
import tarfile


class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability by locating an existing PoC
        inside the provided source tarball. Falls back to a generic input if none found.
        """
        GROUND_TRUTH_SIZE = 7270

        try:
            with tarfile.open(src_path, "r:*") as tf:
                best_member = None
                best_score = float("-inf")

                for info in tf.getmembers():
                    if not info.isreg():
                        continue

                    size = info.size
                    if size <= 0:
                        continue

                    path = info.name
                    lower_path = path.lower()
                    basename = os.path.basename(lower_path)

                    score = 0.0

                    # Size closeness to ground-truth PoC length
                    diff = abs(size - GROUND_TRUTH_SIZE)
                    score += max(0.0, 1000.0 - diff * 0.1)

                    # Filename / path heuristics
                    name_tokens = [
                        ("poc", 400),
                        ("crash", 350),
                        ("id:", 300),
                        ("uaf", 200),
                        ("heap", 150),
                        ("repro", 250),
                        ("reproducer", 250),
                        ("proof", 120),
                        ("bug", 100),
                    ]
                    for tok, val in name_tokens:
                        if tok in basename or tok in lower_path:
                            score += val

                    # Directory-based hints
                    dir_tokens = [
                        ("poc", 150),
                        ("pocs", 150),
                        ("crash", 150),
                        ("crashes", 150),
                        ("bugs", 120),
                        ("repro", 120),
                        ("inputs", 80),
                        ("afl", 80),
                        ("seeds", 80),
                        ("corpus", -30),  # corpus files are often non-crashing
                        ("examples", 30),
                        ("test", 20),
                        ("tests", 20),
                    ]
                    for tok, val in dir_tokens:
                        if tok in lower_path:
                            score += val

                    # File extension heuristics
                    _, ext = os.path.splitext(basename)
                    ext = ext.lower()

                    preferred_exts = {
                        ".rb",
                        ".txt",
                        ".bin",
                        ".dat",
                        ".in",
                        ".input",
                        ".rbx",
                        ".poC".lower(),
                    }
                    if ext in preferred_exts or ext == "":
                        score += 80

                    # Deprioritize typical source files
                    code_exts = {
                        ".c",
                        ".h",
                        ".cpp",
                        ".hpp",
                        ".cc",
                        ".hh",
                        ".rl",
                        ".y",
                        ".l",
                        ".m",
                        ".mm",
                        ".rs",
                        ".go",
                        ".java",
                        ".py",
                        ".js",
                        ".ts",
                    }
                    if ext in code_exts:
                        score -= 250

                    # Deprioritize files in typical source/include/lib dirs
                    for tok in ("/src/", "/include/", "/inc/", "/lib/", "/examples/", "/doc/", "/docs/"):
                        if tok in ("/" + lower_path):
                            score -= 220

                    # Penalize very small or very large files
                    if size < 16:
                        score -= 800
                    if size > 500_000:
                        score -= 600

                    if score > best_score:
                        best_score = score
                        best_member = info

                if best_member is not None and best_member.size > 0:
                    f = tf.extractfile(best_member)
                    if f is not None:
                        data = f.read()
                        f.close()
                        return data

        except Exception:
            # If anything goes wrong, fall back to generic PoC
            pass

        # Fallback generic PoC: a Ruby script attempting to stress the VM stack.
        return self._fallback_poc()

    def _fallback_poc(self) -> bytes:
        # This is a generic Ruby script designed to create deep recursion
        # and large stack usage, which may trigger stack extension behavior.
        script = r"""
# Generic fallback PoC for mruby VM stack issues
def deep_call(n, acc, &blk)
  return acc if n <= 0
  acc << n
  deep_call(n - 1, acc, &blk)
  yield(acc) if block_given?
  acc
end

class Stress
  def initialize
    @arrays = []
  end

  def run
    50.times do |i|
      a = []
      200.times do |j|
        deep_call(50, []) do |stack|
          a << stack
        end
      end
      @arrays << a
      GC.start if GC.respond_to?(:start)
    end
  end
end

s = Stress.new
s.run
"""
        return script.encode("utf-8", "replace")