"""
add_author_metadata.py
=====================
Add author metadata to all project artifacts.

Updates files across:
- reports/final/ (CSVs, JSONs, PDFs, text)
- reports/figures/ (PNG metadata)
- data/processed/ (CSV metadata)
- docs/ (text files)

For each file type:
- Text files: Insert "Author: Diyar Erol" header
- CSV files: Add "# Author: Diyar Erol" comment
- PDF files: Embed author metadata
- Model files (.pkl): Create .meta.json companion

Logs all changes to reports/final/metadata_log.txt
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import shutil
import warnings
warnings.filterwarnings('ignore')

# Try to import PDF libraries
try:
    from PyPDF2 import PdfWriter
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False


def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == 'src':
        return current.parent
    return current


class MetadataLogger:
    """Log all metadata changes."""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.entries = []
        self.counts = {
            'text': 0,
            'csv': 0,
            'json': 0,
            'pdf': 0,
            'pkl': 0,
            'png': 0,
            'skipped': 0
        }
        
        # Initialize log file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("AUTHOR METADATA UPDATE LOG\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Author: Diyar Erol\n")
            f.write("="*70 + "\n\n")
    
    def log_entry(self, file_path, status, file_type, message=""):
        """Log a single file update."""
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file': str(file_path.relative_to(get_project_root())),
            'type': file_type,
            'status': status,
            'message': message
        }
        self.entries.append(entry)
        
        # Write to file immediately
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{entry['timestamp']}] {entry['status']:<10} {entry['file']:<50} ({entry['type']})\n")
            if message:
                f.write(f"  → {message}\n")
    
    def increment_count(self, file_type):
        """Increment file type counter."""
        if file_type in self.counts:
            self.counts[file_type] += 1
    
    def write_summary(self):
        """Write summary to log file."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*70 + "\n")
            f.write("SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Text files updated: {self.counts['text']}\n")
            f.write(f"CSV files updated: {self.counts['csv']}\n")
            f.write(f"JSON files updated: {self.counts['json']}\n")
            f.write(f"PDF files updated: {self.counts['pdf']}\n")
            f.write(f"PKL companions created: {self.counts['pkl']}\n")
            f.write(f"PNG metadata updated: {self.counts['png']}\n")
            f.write(f"Files skipped: {self.counts['skipped']}\n")
            f.write(f"Total: {sum(self.counts.values())}\n")
            f.write("="*70 + "\n")


def add_author_to_text_file(file_path, logger):
    """Add author header to text files (.md, .txt, etc)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if author already exists (case-insensitive)
        if 'author:' in content.lower():
            logger.log_entry(file_path, 'SKIP', 'text', 'Author already present')
            logger.increment_count('skipped')
            return False
        
        # Add author header at the top
        author_header = "Author: Diyar Erol\n"
        new_content = author_header + content
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.log_entry(file_path, 'UPDATE', 'text', 'Author header added')
        logger.increment_count('text')
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'text', str(e))
        return False


def add_author_to_csv_file(file_path, logger):
    """Add author comment to CSV files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if author already exists
        if '# Author:' in content or 'Author:' in content.split('\n')[0]:
            logger.log_entry(file_path, 'SKIP', 'csv', 'Author already present')
            logger.increment_count('skipped')
            return False
        
        # Add author comment at the top
        author_comment = "# Author: Diyar Erol\n"
        new_content = author_comment + content
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.log_entry(file_path, 'UPDATE', 'csv', 'Author comment added')
        logger.increment_count('csv')
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'csv', str(e))
        return False


def add_author_to_json_file(file_path, logger):
    """Add author field to JSON files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if author already exists
        if 'author' in data or 'Author' in data:
            logger.log_entry(file_path, 'SKIP', 'json', 'Author already present')
            logger.increment_count('skipped')
            return False
        
        # Add author field
        data['author'] = 'Diyar Erol'
        data['metadata_updated'] = datetime.now().isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.log_entry(file_path, 'UPDATE', 'json', 'Author field added')
        logger.increment_count('json')
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'json', str(e))
        return False


def add_author_to_pdf(file_path, logger):
    """Add author metadata to PDF files using PyPDF2 or PyMuPDF."""
    try:
        success = False
        
        # Try PyMuPDF first (better for metadata)
        if FITZ_AVAILABLE:
            try:
                pdf = fitz.open(file_path)
                metadata = pdf.metadata
                
                # Check if author already set
                if metadata and metadata.get('author'):
                    logger.log_entry(file_path, 'SKIP', 'pdf', 'Author already present')
                    logger.increment_count('skipped')
                    pdf.close()
                    return False
                
                # Set author metadata
                pdf.set_metadata({
                    'author': 'Diyar Erol',
                    'title': metadata.get('title', file_path.stem) if metadata else file_path.stem,
                    'subject': metadata.get('subject', 'Analysis Report') if metadata else 'Analysis Report',
                    'creator': 'add_author_metadata.py'
                })
                
                pdf.save(file_path)
                pdf.close()
                
                logger.log_entry(file_path, 'UPDATE', 'pdf', 'Author metadata added (PyMuPDF)')
                logger.increment_count('pdf')
                success = True
                
            except Exception as e:
                print(f"[DEBUG] PyMuPDF failed: {e}")
        
        # Fallback to PyPDF2
        if not success and PYPDF2_AVAILABLE:
            try:
                reader = PdfWriter()
                
                # Read existing PDF
                with open(file_path, 'rb') as f:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Check existing metadata
                    if pdf_reader.metadata and pdf_reader.metadata.get('/Author'):
                        logger.log_entry(file_path, 'SKIP', 'pdf', 'Author already present')
                        logger.increment_count('skipped')
                        return False
                    
                    # Copy pages
                    for page in pdf_reader.pages:
                        reader.add_page(page)
                
                # Add author metadata
                reader.add_metadata({
                    '/Author': 'Diyar Erol',
                    '/Creator': 'add_author_metadata.py',
                    '/Subject': 'Analysis Report'
                })
                
                # Write updated PDF
                with open(file_path, 'wb') as f:
                    reader.write(f)
                
                logger.log_entry(file_path, 'UPDATE', 'pdf', 'Author metadata added (PyPDF2)')
                logger.increment_count('pdf')
                success = True
                
            except Exception as e:
                print(f"[DEBUG] PyPDF2 failed: {e}")
        
        if not success:
            logger.log_entry(file_path, 'SKIP', 'pdf', 
                           'No PDF library available (install: pip install PyMuPDF)')
            logger.increment_count('skipped')
            return False
        
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'pdf', str(e))
        return False


def add_metadata_to_pkl(file_path, logger):
    """Create metadata companion JSON for .pkl files."""
    try:
        # Check if companion already exists
        meta_path = file_path.with_suffix('.meta.json')
        
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            if meta.get('author'):
                logger.log_entry(file_path, 'SKIP', 'pkl', 'Metadata companion already exists')
                logger.increment_count('skipped')
                return False
        
        # Create metadata
        metadata = {
            'author': 'Diyar Erol',
            'created': datetime.now().isoformat(),
            'source_file': file_path.name,
            'description': f'Metadata for {file_path.name}',
            'file_size_bytes': file_path.stat().st_size
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.log_entry(file_path, 'UPDATE', 'pkl', f'Metadata companion created: {meta_path.name}')
        logger.increment_count('pkl')
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'pkl', str(e))
        return False


def add_metadata_to_png(file_path, logger):
    """Add author metadata to PNG files."""
    try:
        # PNG metadata requires PIL/Pillow
        try:
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo
        except ImportError:
            logger.log_entry(file_path, 'SKIP', 'png', 
                           'PIL not available (install: pip install Pillow)')
            logger.increment_count('skipped')
            return False
        
        # Load image
        img = Image.open(file_path)
        
        # Check if author already in metadata
        if img.info and img.info.get('Author'):
            logger.log_entry(file_path, 'SKIP', 'png', 'Author already in metadata')
            logger.increment_count('skipped')
            img.close()
            return False
        
        # Create backup
        backup_path = file_path.with_suffix('.backup.png')
        shutil.copy2(file_path, backup_path)
        
        # Add metadata and save
        metadata = PngInfo()
        for key, value in img.info.items():
            if isinstance(value, (str, int, float)):
                metadata.add_text(str(key), str(value))
        
        metadata.add_text('Author', 'Diyar Erol')
        metadata.add_text('Modified', datetime.now().isoformat())
        
        img.save(file_path, pnginfo=metadata)
        img.close()
        
        # Remove backup if successful
        backup_path.unlink()
        
        logger.log_entry(file_path, 'UPDATE', 'png', 'Author metadata added')
        logger.increment_count('png')
        return True
        
    except Exception as e:
        logger.log_entry(file_path, 'ERROR', 'png', str(e))
        return False


def process_directory(directory, logger):
    """Recursively process all files in directory."""
    if not directory.exists():
        print(f"[WARNING] Directory not found: {directory}")
        return
    
    for file_path in directory.rglob('*'):
        if not file_path.is_file():
            continue
        
        suffix = file_path.suffix.lower()
        
        # Text files
        if suffix in ['.md', '.txt']:
            add_author_to_text_file(file_path, logger)
        
        # CSV files
        elif suffix == '.csv':
            add_author_to_csv_file(file_path, logger)
        
        # JSON files
        elif suffix == '.json':
            add_author_to_json_file(file_path, logger)
        
        # PDF files
        elif suffix == '.pdf':
            add_author_to_pdf(file_path, logger)
        
        # Model files
        elif suffix == '.pkl':
            add_metadata_to_pkl(file_path, logger)
        
        # Image files (be cautious)
        elif suffix == '.png':
            # Only for small images (< 10 MB)
            if file_path.stat().st_size < 10 * 1024 * 1024:
                add_metadata_to_png(file_path, logger)
            else:
                logger.log_entry(file_path, 'SKIP', 'png', 'File too large')
                logger.increment_count('skipped')
        
        # YAML/YML files
        elif suffix in ['.yaml', '.yml']:
            add_author_to_text_file(file_path, logger)
        
        # Other files (skip)
        else:
            logger.log_entry(file_path, 'SKIP', 'other', f'Unsupported format: {suffix}')
            logger.increment_count('skipped')


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("ADD AUTHOR METADATA - Attach Diyar Erol as Author")
    print("="*70 + "\n")
    
    root = get_project_root()
    
    # Setup logger
    log_path = root / 'reports' / 'final' / 'metadata_log.txt'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = MetadataLogger(log_path)
    
    # Directories to process
    directories = [
        root / 'reports' / 'final',
        root / 'reports' / 'figures',
        root / 'data' / 'processed',
        root / 'docs',
        root / 'src' / 'reports' / 'final',
        root / 'src' / 'reports' / 'figures',
        root / 'src' / 'data' / 'processed',
        root / 'src' / 'docs'
    ]
    
    print("[1/3] Checking PDF library availability...")
    if FITZ_AVAILABLE:
        print("[OK] PyMuPDF (fitz) available")
    elif PYPDF2_AVAILABLE:
        print("[OK] PyPDF2 available")
    else:
        print("[WARNING] No PDF library found. PDF metadata will be skipped.")
        print("Install with: pip install PyMuPDF  (recommended)")
        print("          or: pip install PyPDF2")
    
    print("\n[2/3] Processing files...")
    for directory in directories:
        if directory.exists():
            print(f"  Scanning: {directory.relative_to(root)}/")
            process_directory(directory, logger)
    
    # Write summary
    print("\n[3/3] Writing summary...")
    logger.write_summary()
    
    # Print summary
    print("\n" + "="*70)
    print("METADATA UPDATE SUMMARY")
    print("="*70)
    print(f"Text files updated:      {logger.counts['text']}")
    print(f"CSV files updated:       {logger.counts['csv']}")
    print(f"JSON files updated:      {logger.counts['json']}")
    print(f"PDF files updated:       {logger.counts['pdf']}")
    print(f"PKL companions created:  {logger.counts['pkl']}")
    print(f"PNG metadata updated:    {logger.counts['png']}")
    print(f"Files skipped:           {logger.counts['skipped']}")
    print(f"Total processed:         {sum(logger.counts.values())}")
    print("="*70)
    
    print(f"\nLog file: {log_path}")
    print(f"\n[SUCCESS] Metadata update completed!\n")


if __name__ == "__main__":
    main()
