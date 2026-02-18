"""
================================================================================
FILE READER MODULE
================================================================================
Modul untuk membaca file TXT dan PDF

Author: Member 1 - NLP & Backend Engineer
Phase: 1 (Foundation & Setup)

Functions:
    - read_txt(file_path): Membaca file .txt
    - read_pdf(file_path): Membaca file .pdf  
    - read_file(file_path): Universal reader (auto-detect format)

Usage:
    from modules.file_reader import read_file
    content = read_file('data/sample.txt')
================================================================================
"""

import os
from typing import Optional
import PyPDF2
import re


def fix_pdf_spacing(text: str) -> str:
    """
    Fix spacing issues in PDF extracted text where words concatenate without spaces.
    Uses capitalization pattern to detect word boundaries.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Text with proper spacing
        
    Example:
        >>> fix_pdf_spacing("ThorfinnKarlsefniadalahpejuang")
        "Thorfinn Karlsefni adalah pejuang"
    """
    # CLEANUP FIRST: Remove excessive spaces before single capital letters
    # Pattern: " X " where X is single capital -> "X "
    text = re.sub(r'\s([A-Z])\s', r'\1 ', text)
    
    # PATTERN 1: lowercase followed by uppercase (e.g., "manusiaViking" -> "manusia Viking")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # PATTERN 2: Multiple uppercase followed by lowercase (e.g., "VIKINGadalah" -> "VIKING adalah")
    text = re.sub(r'([A-Z]+)([a-z])', r'\1 \2', text)
    
    # PATTERN 3: Number followed by letter (e.g., "2024Thorfinn" -> "2024 Thorfinn")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # PATTERN 4: Letter followed by number (e.g., "Thorfinn2024" -> "Thorfinn 2024")  
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # Clean up multiple spaces aggressively
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def read_txt(file_path: str) -> Optional[str]:
    """
    Membaca konten dari file .txt
    
    Args:
        file_path (str): Path ke file .txt
        
    Returns:
        Optional[str]: Konten file sebagai string, atau None jika ada error
        
    Example:
        >>> content = read_txt('data/artikel.txt')
        >>> print(content[:100])
    """
    try:
        # Validasi: Cek apakah file ada
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
        # Validasi: Cek ekstensi file
        if not file_path.endswith('.txt'):
            raise ValueError("File harus berformat .txt")
        
        # Baca file dengan encoding UTF-8
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Validasi: Cek apakah file kosong
        if not content.strip():
            raise ValueError("File kosong")
        
        # Log sukses
        print(f"[OK] Berhasil membaca file TXT: {file_path}")
        print(f" Panjang konten: {len(content)} karakter")
        
        return content
        
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] Error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error tidak terduga saat membaca TXT: {e}")
        return None


def read_pdf(file_path: str) -> Optional[str]:
    """
    Membaca konten dari file .pdf
    
    Args:
        file_path (str): Path ke file .pdf
        
    Returns:
        Optional[str]: Teks yang diekstrak dari PDF, atau None jika ada error
        
    Example:
        >>> content = read_pdf('data/paper.pdf')
        >>> print(f"Jumlah kata: {len(content.split())}")
    """
    try:
        # Validasi: Cek apakah file ada
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
        # Validasi: Cek ekstensi file
        if not file_path.endswith('.pdf'):
            raise ValueError("File harus berformat .pdf")
        
        # SOLUSI 3: Try pdfplumber first (more reliable)
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF tidak memiliki halaman")
                
                # Extract text from all pages
                text_content = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                
                full_text = "\n".join(text_content)
                
                if not full_text.strip():
                    raise ValueError("Tidak ada teks yang bisa diekstrak dari PDF")
                
                # Log sukses
                print(f"[OK] Berhasil membaca file PDF (pdfplumber): {file_path}")
                print(f" Total halaman: {len(pdf.pages)}")
                print(f" Panjang konten: {len(full_text)} karakter")
                
                return full_text
                
        except ImportError:
            # Fallback to PyPDF2 if pdfplumber not available
            print("[INFO] pdfplumber not available, using PyPDF2...")
            pass
        
        # FALLBACK: PyPDF2 extraction with spacing fix
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Validasi: Cek apakah PDF punya halaman
            if len(pdf_reader.pages) == 0:
                raise ValueError("PDF tidak memiliki halaman")
            
            # Ekstrak teks dari semua halaman
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            
            # Gabungkan semua halaman
            full_text = "\n".join(text_content)
            
            # Validasi: Cek apakah ada teks yang diekstrak
            if not full_text.strip():
                raise ValueError("Tidak ada teks yang bisa diekstrak dari PDF")
            
            # FIX: Apply spacing fix untuk handle kata yang nempel
            full_text = fix_pdf_spacing(full_text)
            
            # Log sukses
            print(f"[OK] Berhasil membaca file PDF (PyPDF2 + fix): {file_path}")
            print(f" Total halaman: {len(pdf_reader.pages)}")
            print(f" Panjang konten: {len(full_text)} karakter")
            
            return full_text
            
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] Error: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error tidak terduga saat membaca PDF: {e}")
        return None


def read_file(file_path: str) -> Optional[str]:
    """
    Universal file reader yang otomatis mendeteksi tipe file
    
    Args:
        file_path (str): Path ke file
        
    Returns:
        Optional[str]: Konten file, atau None jika ada error
        
    Example:
        >>> # Bisa untuk TXT atau PDF
        >>> content = read_file('data/document.pdf')
        >>> content = read_file('data/article.txt')
    """
    if file_path.endswith('.txt'):
        return read_txt(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    else:
        print(f"[ERROR] Format file tidak didukung. Hanya .txt dan .pdf yang didukung.")
        return None


# ============================================================================
# TESTING CODE (untuk development)
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("FILE READER MODULE - TEST")
    print("="*80)
    print()
    
    # Test dengan file yang akan dibuat
    print(" CATATAN: Pastikan ada file sample di folder data/")
    print("   - data/sample.txt")
    print("   - data/sample.pdf (opsional)")
    print()
    
    # Test TXT
    print("1️⃣  Testing TXT file reader:")
    print("-"*80)
    txt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.txt')
    txt_content = read_txt(txt_path)
    if txt_content:
        print(f"Preview: {txt_content[:200]}...")
    print()
    
    # Test universal reader
    print("2️⃣  Testing universal file reader:")
    print("-"*80)
    content = read_file(txt_path)
    if content:
        print(f"[OK] Universal reader berhasil!")
