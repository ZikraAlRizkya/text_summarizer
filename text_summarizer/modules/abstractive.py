"""
================================================================================
ABSTRACTIVE SUMMARIZATION MODULE
================================================================================
Integrasi dengan Google Gemini API untuk abstractive summarization

Author: Member 1 - NLP & Backend Engineer
Phase: 2 (Core Summarization)

Perbedaan Extractive vs Abstractive:
    - Extractive: Memilih kalimat penting dari teks asli (copy-paste)
    - Abstractive: Generate kalimat baru yang merangkum isi teks (paraphrase)

Features:
    - Integrasi Gemini API
    - Prompt engineering untuk summary berkualitas
    - Multiple summary styles (concise, detailed, bullet)

Classes:
    - GeminiSummarizer: Class utama untuk abstractive summarization

Usage:
    from modules.abstractive import GeminiSummarizer
    
    summarizer = GeminiSummarizer()
    result = summarizer.summarize(text, max_sentences=5)
    print(result['summary'])
================================================================================
"""

import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Optional, Dict

# Load environment variables dari .env file
load_dotenv()


class GeminiSummarizer:
    """
    Abstractive summarization menggunakan Google Gemini API
    
    Gemini adalah Large Language Model (LLM) dari Google yang dapat
    memahami dan menghasilkan teks dengan kualitas tinggi.
    """
    
    def __init__(self, model_name='models/gemini-2.5-flash'):
        """
        Inisialisasi Gemini summarizer
        
        Args:
            model_name (str): Nama model Gemini yang digunakan
                            (default: 'models/gemini-2.5-flash')
                            
        Note:
            Model yang tersedia (Februari 2026):
            - 'models/gemini-2.5-flash' (recommended - fast)
            - 'models/gemini-2.5-pro' (high quality)
            - 'models/gemini-flash-latest' (always latest)
            
            Deprecated models (NOT AVAILABLE):
            - gemini-1.5-flash, gemini-1.5-pro, gemini-pro
        
        Raises:
            ValueError: Jika API key tidak ditemukan
        """
        # Ambil API key dari environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY tidak ditemukan!\n"
                "Pastikan file .env ada dan berisi:\n"
                "GEMINI_API_KEY=your_api_key_here"
            )
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        print(f"[OK] Gemini Summarizer initialized: {model_name}")
    
    # ========================================================================
    # PROMPT ENGINEERING
    # ========================================================================
    
    def create_summary_prompt(
        self, 
        text: str, 
        max_sentences: int = 5,
        style: str = "concise"
    ) -> str:
        """
        Membuat prompt yang optimal untuk summarization
        
        Prompt Engineering Tips:
            - Clear instruction: Jelaskan task dengan jelas
            - Context: Berikan konteks tentang teks
            - Format: Spesifikasikan format output yang diinginkan
            - Constraints: Batasan seperti panjang summary
            - Quality criteria: Kriteria kualitas (concise, coherent, dll)
        
        Args:
            text (str): Teks yang akan dirangkum
            max_sentences (int): Maximum kalimat dalam summary
            style (str): Style summary ('concise', 'detailed', 'bullet')
            
        Returns:
            str: Formatted prompt untuk Gemini
        """
        # Definisi style instructions
        style_instructions = {
            'concise': "Buat ringkasan yang singkat dan padat (concise).",
            'detailed': "Buat ringkasan yang komprehensif dan detail.",
            'bullet': "Buat ringkasan dalam bentuk bullet points."
        }
        
        instruction = style_instructions.get(style, style_instructions['concise'])
        
        # Construct prompt
        prompt = f"""Kamu adalah expert text summarizer. Tugas kamu adalah membuat ringkasan berkualitas tinggi dari teks berikut.

Instruksi:
- {instruction}
- Maksimal {max_sentences} kalimat
- Tangkap main ideas dan key points
- Gunakan bahasa yang clear dan coherent
- Jangan include detail yang tidak perlu
- Maintain factual accuracy

Teks yang akan dirangkum:
{text}

Ringkasan:"""
        
        return prompt
    
    # ========================================================================
    # MAIN SUMMARIZATION FUNCTION
    # ========================================================================
    
    def summarize(
        self, 
        text: str, 
        max_sentences: int = 5,
        style: str = "concise",
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate abstractive summary menggunakan Gemini API
        
        Args:
            text (str): Input text yang akan dirangkum
            max_sentences (int): Maximum kalimat dalam summary (default: 5)
            style (str): Summary style - 'concise', 'detailed', atau 'bullet'
            temperature (float): Kreativitas model (0.0-1.0)
                               - 0.0: Sangat deterministik
                               - 1.0: Sangat kreatif/random
                               - Recommended: 0.3 untuk summarization
            
        Returns:
            dict: Dictionary berisi:
                - summary: String summary yang dihasilkan
                - model: Nama model yang digunakan
                - style: Style yang digunakan
                - success: Boolean apakah berhasil atau tidak
                - error: Error message (jika ada)
                
        Example:
            >>> summarizer = GeminiSummarizer()
            >>> result = summarizer.summarize(
            ...     text="Long article...",
            ...     max_sentences=3,
            ...     style="concise"
            ... )
            >>> if result['success']:
            ...     print(result['summary'])
        """
        try:
            print(f" Generating abstractive summary dengan Gemini...")
            print(f"   Model: {self.model_name}")
            print(f"   Style: {style}")
            print(f"   Max sentences: {max_sentences}")
            
            # STEP 1: Create optimized prompt
            prompt = self.create_summary_prompt(text, max_sentences, style)
            
            # STEP 2: Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,      # Kontrolkreativitas
                max_output_tokens=1024,      # Max panjang output
            )
            
            # STEP 3: Generate summary
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # STEP 4: Extract summary dari response
            summary = response.text.strip()
            
            # Log hasil
            print(f"   [OK] Summary berhasil di-generate!")
            print(f"    Panjang summary: {len(summary)} karakter")
            
            return {
                'summary': summary,
                'model': self.model_name,
                'style': style,
                'success': True
            }
            
        except Exception as e:
            # Handle error
            print(f"   [ERROR] Error saat generate summary: {e}")
            return {
                'summary': "",
                'model': self.model_name,
                'style': style,
                'success': False,
                'error': str(e)
            }
    
    # ========================================================================
    # COMPARISON FUNCTION
    # ========================================================================
    
    def compare_summaries(
        self, 
        text: str, 
        extractive_summary: str
    ) -> Dict:
        """
        Generate abstractive summary dan compare dengan extractive summary
        
        Args:
            text (str): Original text
            extractive_summary (str): Extractive summary untuk comparison
            
        Returns:
            dict: Comparison results
        """
        # Generate abstractive summary
        abstractive_result = self.summarize(text)
        
        if not abstractive_result['success']:
            return abstractive_result
        
        abstractive_summary = abstractive_result['summary']
        
        return {
            'original_text': text,
            'extractive_summary': extractive_summary,
            'abstractive_summary': abstractive_summary,
            'extractive_length': len(extractive_summary),
            'abstractive_length': len(abstractive_summary),
            'model': self.model_name
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def abstractive_summary_gemini(
    text: str, 
    max_sentences: int = 5,
    style: str = "concise"
) -> str:
    """
    Fungsi convenience untuk generate abstractive summary
    
    Args:
        text (str): Input text
        max_sentences (int): Maximum kalimat
        style (str): Summary style
        
    Returns:
        str: Summary text (empty string jika error)
    """
    summarizer = GeminiSummarizer()
    result = summarizer.summarize(text, max_sentences, style)
    return result['summary'] if result['success'] else ""


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ABSTRACTIVE SUMMARIZATION (Gemini) - TEST")
    print("="*80)
    print()
    
    # Sample text
    sample_text = """
    Artificial intelligence has made significant advancements in recent years.
    Machine learning algorithms can now process vast amounts of data efficiently.
    Deep learning techniques have revolutionized computer vision and natural language processing.
    Neural networks are inspired by the structure of the human brain.
    AI systems can now perform tasks that were once thought to require human intelligence.
    Self-driving cars use AI to navigate roads and avoid obstacles.
    AI assistants like Siri and Alexa have become commonplace in homes worldwide.
    The future of AI holds both exciting possibilities and important challenges.
    Ethical considerations are crucial as AI becomes more powerful and widespread.
    Researchers continue to push the boundaries of what AI can achieve.
    """
    
    print(" ORIGINAL TEXT:")
    print("-"*80)
    print(sample_text.strip())
    print()
    
    # Test summarization dengan berbagai style
    try:
        summarizer = GeminiSummarizer()
        
        styles = ['concise', 'detailed', 'bullet']
        
        for style in styles:
            print(f"\n{'='*80}")
            print(f"STYLE: {style.upper()}")
            print(f"{'='*80}")
            
            result = summarizer.summarize(
                sample_text, 
                max_sentences=3, 
                style=style
            )
            
            if result['success']:
                print(f"\n{result['summary']}")
            else:
                print(f"\n[ERROR] Error: {result.get('error', 'Unknown error')}")
                
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\n💡 Solusi:")
        print("   1. Pastikan file .env ada di root project")
        print("   2. File .env harus berisi: GEMINI_API_KEY=your_key_here")
        print("   3. Dapatkan API key dari: https://makersuite.google.com/app/apikey")
