"""
================================================================================
MAIN CLI - TEXT SUMMARIZATION TOOL
================================================================================
Command-line interface yang mengintegrasikan semua module untuk complete
summarization pipeline

Author: Member 1 - NLP & Backend Engineer
Phase: 2 (Core Summarization)

Pipeline:
    1. Read file (TXT/PDF)
    2. Preprocessing (cleaning, tokenization)
    3. Extractive summarization (TextRank)
    4. Abstractive summarization (Gemini)
    5. Display & save results

Usage:
    python main.py data/sample.txt
    python main.py data/document.pdf --ratio 0.4
    python main.py data/article.txt --sentences 5 --style detailed
================================================================================
"""

import argparse
import os
import sys
from datetime import datetime

# Add modules to path
sys.path.append(os.path.dirname(__file__))

from modules.file_reader import read_file
from modules.preprocessing import TextPreprocessor
from modules.extractive import TextRankSummarizer
from modules.abstractive import GeminiSummarizer


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    """Print section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)


def print_section(title):
    """Print sub-section"""
    print(f"\n{title}")
    print("-"*80)


def save_summary(original_text, extractive_summary, abstractive_summary, output_file=None):
    """
    Menyimpan hasil summary ke file
    
    Args:
        original_text (str): Teks asli
        extractive_summary (str): Extractive summary
        abstractive_summary (str): Abstractive summary
        output_file (str): Path output file (optional)
        
    Returns:
        str: Path file yang disimpan
    """
    # Generate filename jika tidak dispesifikasikan
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/summary_{timestamp}.txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TEXT SUMMARIZATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("📄 ORIGINAL TEXT:\n")
        f.write("-"*80 + "\n")
        f.write(original_text + "\n\n")
        
        f.write("📝 EXTRACTIVE SUMMARY (TextRank):\n")
        f.write("-"*80 + "\n")
        f.write(extractive_summary + "\n\n")
        
        f.write("🤖 ABSTRACTIVE SUMMARY (Gemini AI):\n")
        f.write("-"*80 + "\n")
        f.write(abstractive_summary + "\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tool: Text Summarization - Member 1\n")
    
    print(f"[OK] Summary disimpan ke: {output_file}")
    return output_file


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main CLI function
    
    Menjalankan complete summarization pipeline:
        1. Parse arguments
        2. Read file
        3. Preprocess
        4. Extractive summarization
        5. Abstractive summarization
        6. Display & save results
    """
    
    # ========================================================================
    # SETUP ARGUMENT PARSER
    # ========================================================================
    
    parser = argparse.ArgumentParser(
        description='🤖 Text Summarization Tool - TextRank + Gemini AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📖 Examples:
  python main.py data/sample.txt
  python main.py data/document.pdf --ratio 0.4
  python main.py data/article.txt --sentences 5 --output results/summary.txt
  python main.py data/paper.pdf --style detailed --no-save

📝 Supported file formats: .txt, .pdf
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'file', 
        type=str, 
        help='Path ke input file (.txt atau .pdf)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--ratio', 
        type=float, 
        default=0.3, 
        help='Rasio extractive summary (default: 0.3 = 30%%)'
    )
    
    parser.add_argument(
        '--sentences', 
        type=int, 
        default=None,
        help='Jumlah kalimat dalam extractive summary (override ratio)'
    )
    
    parser.add_argument(
        '--style', 
        type=str, 
        default='concise',
        choices=['concise', 'detailed', 'bullet'],
        help='Style abstractive summary (default: concise)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Path output file (default: auto-generate di folder output/)'
    )
    
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Jangan save summary ke file'
    )
    
    parser.add_argument(
        '--language',
        '--lang',
        type=str,
        default='english',
        choices=['english', 'indonesian', 'id'],
        help='Bahasa teks input untuk stopwords (default: english)'
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    print_header("TEXT SUMMARIZATION TOOL")
    print(f"{'Member 1: NLP & Backend Engineer'.center(80)}")
    print(f"{'TextRank + Gemini AI'.center(80)}")
    
    # ========================================================================
    # STEP 1: READ FILE
    # ========================================================================
    
    print_section("[1/4] Reading File")
    print(f"File: {args.file}")
    
    text = read_file(args.file)
    if text is None:
        print("\n[ERROR] Failed to read file. Exiting.")
        return
    
    print(f"[OK] File berhasil dibaca: {len(text)} karakter")
    
    # ========================================================================
    # STEP 2: PREPROCESSING
    # ========================================================================
    
    print_section("[2/4] Preprocessing Text")
    
    # Initialize preprocessor with specified language
    preprocessor = TextPreprocessor(language=args.language)
    preprocessed = preprocessor.preprocess_text(text)
    sentences = preprocessed['sentences']
    
    print(f"[OK] Text berhasil dipreprocess:")
    print(f"   - {preprocessed['num_sentences']} kalimat")
    print(f"   - {preprocessed['num_words']} kata")
    
    # ========================================================================
    # STEP 3: EXTRACTIVE SUMMARIZATION
    # ========================================================================
    
    print_section("[3/4] Extractive Summarization (TextRank)")
    
    extractive_summarizer = TextRankSummarizer()
    extractive_result = extractive_summarizer.summarize(
        sentences=sentences,
        num_sentences=args.sentences,
        ratio=args.ratio
    )
    extractive_summary = extractive_result['summary']
    
    print(f"[OK] Extractive summary generated:")
    print(f"   - {extractive_result['num_sentences']} kalimat dipilih")
    print(f"   - Compression ratio: {extractive_result['compression_ratio']:.2%}")
    
    # ========================================================================
    # STEP 4: ABSTRACTIVE SUMMARIZATION
    # ========================================================================
    
    print_section("[4/4] Abstractive Summarization (Gemini AI)")
    
    try:
        abstractive_summarizer = GeminiSummarizer()
        abstractive_result = abstractive_summarizer.summarize(
            text=text,
            max_sentences=extractive_result['num_sentences'],
            style=args.style
        )
        
        if abstractive_result['success']:
            abstractive_summary = abstractive_result['summary']
            print(f"[OK] Abstractive summary generated")
        else:
            abstractive_summary = "(Abstractive summary tidak tersedia)"
            print(f"[WARNING] {abstractive_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"[WARNING] Tidak bisa generate abstractive summary: {e}")
        abstractive_summary = "(Abstractive summary tidak tersedia - Error API)"
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print_header("HASIL SUMMARIZATION")
    
    # Original text preview
    print_section("ORIGINAL TEXT (Preview)")
    preview_length = min(500, len(text))
    print(text[:preview_length] + ("..." if len(text) > preview_length else ""))
    
    # Extractive summary
    print_section("EXTRACTIVE SUMMARY (TextRank)")
    print(extractive_summary)
    
    # Abstractive summary
    print_section("ABSTRACTIVE SUMMARY (Gemini AI)")
    print(abstractive_summary)
    
    # Statistics
    print_section("STATISTICS")
    print(f"Original:           {len(text)} chars, {len(sentences)} sentences")
    print(f"Extractive:         {len(extractive_summary)} chars, {extractive_result['num_sentences']} sentences")
    print(f"Abstractive:        {len(abstractive_summary)} chars")
    print(f"Compression ratio:  {extractive_result['compression_ratio']:.2%}")
    
    # ========================================================================
    # SAVE TO FILE
    # ========================================================================
    
    if not args.no_save:
        print_section("SAVING RESULTS")
        save_summary(text, extractive_summary, abstractive_summary, args.output)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    print_header("SUMMARIZATION COMPLETE!")
    print(f"{'Terima kasih telah menggunakan Text Summarization Tool'.center(80)}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Process interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        print("Please check your input and try again.")
