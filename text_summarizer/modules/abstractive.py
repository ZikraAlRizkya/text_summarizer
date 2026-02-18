# Modul abstractive summarization — generate ringkasan baru pakai Gemini API
# (beda sama extractive yang copy-paste kalimat asli, ini bikin kalimat baru)

import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Optional, Dict

# Load environment variables dari .env file
load_dotenv()


class GeminiSummarizer:
    # Class utama untuk generate summary pakai Gemini LLM

    def __init__(self, model_name='models/gemini-2.5-flash'):
        # Inisialisasi: ambil API key dari .env, lalu connect ke model Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY tidak ditemukan!\n"
                "Pastikan file .env ada dan berisi:\n"
                "GEMINI_API_KEY=your_api_key_here"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        print(f"[OK] Gemini Summarizer initialized: {model_name}")
    
    def create_summary_prompt(
        self, 
        text: str, 
        max_sentences: int = 5,
        style: str = "concise"
    ) -> str:
        # Bikin prompt yang dikirim ke Gemini — atur style (concise/detailed/bullet)
        # dan batasan jumlah kalimat output
        style_instructions = {
            'concise': "Buat ringkasan yang singkat dan padat (concise).",
            'detailed': "Buat ringkasan yang komprehensif dan detail.",
            'bullet': "Buat ringkasan dalam bentuk bullet points."
        }
        
        instruction = style_instructions.get(style, style_instructions['concise'])
        
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
    
    def summarize(
        self, 
        text: str, 
        max_sentences: int = 5,
        style: str = "concise",
        temperature: float = 0.3
    ) -> Dict:
        # Fungsi utama: kirim teks ke Gemini dan return hasilnya sebagai dict
        # temperature rendah (0.3) biar outputnya konsisten, bukan random
        try:
            print(f" Generating abstractive summary dengan Gemini...")
            print(f"   Model: {self.model_name}")
            print(f"   Style: {style}")
            print(f"   Max sentences: {max_sentences}")
            
            prompt = self.create_summary_prompt(text, max_sentences, style)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            summary = response.text.strip()
            
            print(f"   [OK] Summary berhasil di-generate!")
            print(f"    Panjang summary: {len(summary)} karakter")
            
            return {
                'summary': summary,
                'model': self.model_name,
                'style': style,
                'success': True
            }
            
        except Exception as e:
            print(f"   [ERROR] Error saat generate summary: {e}")
            return {
                'summary': "",
                'model': self.model_name,
                'style': style,
                'success': False,
                'error': str(e)
            }
    
    def compare_summaries(
        self, 
        text: str, 
        extractive_summary: str
    ) -> Dict:
        # Bandingin hasil abstractive vs extractive — return keduanya sekaligus
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


# Shortcut function — langsung return string summary tanpa perlu bikin object dulu
def abstractive_summary_gemini(
    text: str, 
    max_sentences: int = 5,
    style: str = "concise"
) -> str:
    summarizer = GeminiSummarizer()
    result = summarizer.summarize(text, max_sentences, style)
    return result['summary'] if result['success'] else ""


# Blok testing — jalan kalau file ini dirun langsung (bukan di-import)
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
