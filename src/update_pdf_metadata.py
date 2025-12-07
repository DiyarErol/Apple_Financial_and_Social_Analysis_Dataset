"""
update_pdf_metadata.py
======================
Update PDF metadata with comprehensive author and repository information.
"""

import sys
from pathlib import Path
from datetime import datetime

def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current

def update_pdf_metadata_fitz(file_path):
    """Update PDF metadata using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        return False, "PyMuPDF not installed"
    
    try:
        pdf = fitz.open(file_path)
        metadata = pdf.metadata
        
        # Set comprehensive metadata
        pdf.set_metadata({
            'author': 'Diyar Erol',
            'subject': 'Apple Financial and Social Analysis Dataset (2015–2025)',
            'keywords': 'Finance, Apple, Dataset, Diyar Erol, Machine Learning',
            'producer': 'Apple Dataset Pipeline by Diyar Erol',
            'creator': 'update_pdf_metadata.py',
            'title': metadata.get('title', file_path.stem) if metadata else file_path.stem
        })
        
        pdf.save(file_path)
        pdf.close()
        
        return True, "Updated with PyMuPDF"
        
    except Exception as e:
        return False, f"PyMuPDF error: {str(e)}"

def update_pdf_metadata_pypdf2(file_path):
    """Update PDF metadata using PyPDF2."""
    try:
        from PyPDF2 import PdfWriter, PdfReader
    except ImportError:
        return False, "PyPDF2 not installed"
    
    try:
        # Read existing PDF
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            writer = PdfWriter()
            
            # Copy all pages
            for page in pdf_reader.pages:
                writer.add_page(page)
        
        # Add comprehensive metadata
        writer.add_metadata({
            '/Author': 'Diyar Erol',
            '/Subject': 'Apple Financial and Social Analysis Dataset (2015–2025)',
            '/Keywords': 'Finance, Apple, Dataset, Diyar Erol, Machine Learning',
            '/Producer': 'Apple Dataset Pipeline by Diyar Erol',
            '/Creator': 'update_pdf_metadata.py'
        })
        
        # Write updated PDF
        with open(file_path, 'wb') as f:
            writer.write(f)
        
        return True, "Updated with PyPDF2"
        
    except Exception as e:
        return False, f"PyPDF2 error: {str(e)}"

def main():
    """Update all PDF files in the project."""
    root = get_project_root()
    
    print("\n" + "="*70)
    print("UPDATE PDF METADATA - Add Author and Repository Information")
    print("="*70 + "\n")
    
    # Find all PDF files
    pdf_files = list(root.rglob('*.pdf'))
    
    if not pdf_files:
        print("[INFO] No PDF files found in project.\n")
        return
    
    print(f"[INFO] Found {len(pdf_files)} PDF file(s):\n")
    
    updated_count = 0
    
    for pdf_file in pdf_files:
        rel_path = pdf_file.relative_to(root)
        print(f"[*] {rel_path}...", end=" ")
        
        # Try PyMuPDF first (better)
        success, message = update_pdf_metadata_fitz(pdf_file)
        
        if not success:
            # Try PyPDF2
            success, message = update_pdf_metadata_pypdf2(pdf_file)
        
        if success:
            print(f"✓ {message}")
            updated_count += 1
        else:
            print(f"✗ {message}")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {updated_count}/{len(pdf_files)} PDF files updated")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
