"""
================================================================================
PREPROCESSING MODULE
================================================================================
Bersihin teks. Dari raw jadi siap olah.

Author: Member 1 - NLP & Backend Engineer
Phase: 2 (Core Summarization)

Features:
    - Text cleaning (ilangin URL, email, char aneh)
    - Case normalization (jadiin lowercase)
    - Tokenization (pecah kalimat & kata)
    - Stopword removal (buang kata umum)

Classes:
    - TextPreprocessor: Class buat preprocessing

Usage:
    from modules.preprocessing import TextPreprocessor
    
    processor = TextPreprocessor()
    result = processor.preprocess_text(text)
    print(result['sentences'])  # List kalimat
    print(result['words'])      # List kata
================================================================================
"""

import re
import string
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("[DOWNLOAD]  Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("[DOWNLOAD]  Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Class buat preprocessing
    
    Nanganin:
        1. Text cleaning
        2. Tokenization
        3. Stopword removal
    """
    
    def __init__(self, language='english'):
        """
        Init class
        
        Args:
            language (str): Bahasa buat stopwords 
                          Options: 'english', 'indonesian', 'id'
                          (default: 'english')
        """
        # Normalize language code
        if language in ['indonesian', 'id', 'indonesia']:
            language = 'indonesian'
        
        self.language = language
        
        # Load stopwords (NLTK atau fallback manual)
        try:
            self.stop_words = set(stopwords.words(language))
            print(f"   [OK] Loaded {len(self.stop_words)} {language} stopwords from NLTK")
        except:
            # Fallback: Indonesian stopwords manual
            if language == 'indonesian':
                self.stop_words = self._get_indonesian_stopwords()
                print(f"   [OK] Loaded {len(self.stop_words)} Indonesian stopwords (manual)")
            else:
                print(f"   [WARNING] Warning: Stopwords untuk '{language}' tidak tersedia")
                self.stop_words = set()
    
    def _get_indonesian_stopwords(self) -> set:
        """
        List stopwords Indo (manual list)
        
        Returns:
            set: Set stopwords
        """
        # Comprehensive Indonesian stopwords
        indonesian_stops = [
            # Common words
            'yang', 'dan', 'di', 'dari', 'untuk', 'dengan', 'pada', 'adalah',
            'ini', 'itu', 'dalam', 'akan', 'atau', 'oleh', 'ke', 'sebagai',
            
            # Pronouns
            'saya', 'kami', 'kita', 'kamu', 'anda', 'dia', 'mereka', 'nya',
            
            # Auxiliary verbs
            'telah', 'sudah', 'dapat', 'bisa', 'harus', 'akan', 'boleh',
            
            # Conjunctions & prepositions
            'juga', 'karena', 'bahwa', 'bahwa', 'seperti', 'antara', 'terhadap',
            'atas', 'bawah', 'kepada', 'tentang', 'hingga', 'sampai',
            
            # Common adjectives/adverbs  
            'tidak', 'sangat', 'lebih', 'paling', 'sekali', 'sering', 'selalu',
            'ada', 'semua', 'setiap', 'beberapa', 'banyak', 'sedikit',
            
            # Question words
            'apa', 'siapa', 'kapan', 'dimana', 'mengapa', 'bagaimana',
            
            # Time markers
            'saat', 'ketika', 'waktu', 'sekarang', 'nanti', 'dulu', 'kemudian',
            
            # Articles & demonstratives
            'sebuah', 'seorang', 'suatu', 'para', 'sang', 'si',
            'tersebut', 'tadi', 'kini', 'begitu', 'begini',
        ]
        return set(indonesian_stops)
    
    # ========================================================================
    # TEXT CLEANING FUNCTIONS
    # ========================================================================
    
    def remove_urls(self, text: str) -> str:
        """
        Buang URL.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text bersih
        """
        # Pattern untuk mendeteksi URL (http, https, www)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Buang email.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text bersih
        """
        # Pattern untuk mendeteksi email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def fix_stuck_words(self, text: str) -> str:
        """
        Benerin kata nempel (biasanya dari PDF).
        Pake CamelCase buat deteksi.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text udah bener spasinya
            
        Example:
            >>> preprocessor.fix_stuck_words("ThorfinnKarlsefni")
            "Thorfinn Karlsefni"
        """
        # Kasih spasi sebelum huruf kapital yang nempel sama huruf kecil
        # Contoh: "ThorfinnKarlsefni" -> "Thorfinn Karlsefni"
        fixed_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Kasih spasi antara angka dan huruf yang nempel
        fixed_text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', fixed_text)
        fixed_text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', fixed_text)
        
        return fixed_text
    
    def fix_punctuation_spacing(self, text: str) -> str:
        """
        Kasih spasi abis titik biar kalimat kepisah bener.
        Penting buat tokenizing.
        
        Example:
            >>> "Hello.World" -> "Hello. World"
        """
        # Pisahkan titik/tanda tanya/tanda seru dengan huruf kapital berikutnya
        fixed_text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
        return fixed_text
    
    def remove_special_chars(self, text: str, keep_punctuation=True) -> str:
        """
        Buang char aneh-aneh (tapi keep tanda baca penting).
        
        Args:
            text (str): Input text
            keep_punctuation (bool): Simpan tanda baca (. ! ? ,) buat kalimat
            
        Returns:
            str: Text bersih
        """
        # STEP 1: Normalize Unicode whitespace variants ke regular space
        # Ini fix masalah non-breaking space (\u00A0), thin space (\u2009), dll
        # yang sering ada di PDF dan tidak dikenali oleh \s di regex
        text = text.replace('\u00A0', ' ')  # non-breaking space
        text = text.replace('\u2009', ' ')  # thin space
        text = text.replace('\u200B', ' ')  # zero-width space
        text = text.replace('\u202F', ' ')  # narrow no-break space
        text = text.replace('\u2002', ' ')  # en space
        text = text.replace('\u2003', ' ')  # em space
        
        # STEP 2: Replace parentheses and brackets with space first
        # to prevent word concatenation (e.g., "Goals (xG)" -> "Goals xG" not "GoalsxG")
        text = re.sub(r'[(){}\[\]]', ' ', text)
        
        if keep_punctuation:
            # Simpan tanda baca untuk sentence structure
            # Support Indonesian characters (keep accented letters if any)
            pattern = r'[^a-zA-Z0-9\s\.\!\?\,\-\'\"\/]'
        else:
            # Hapus semua kecuali huruf dan angka
            pattern = r'[^a-zA-Z0-9\s]'
        
        # STEP 3: Remove remaining special characters
        text = re.sub(pattern, '', text)
        
        # STEP 4: Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Rapiin spasi (biar ga double).
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text rapi
        """
        # Ganti multiple spaces dengan single space
        text = re.sub(r'\s+', ' ', text)
        # Hapus leading/trailing whitespace
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Pipeline pembersih utama.
        
        Proses:
            0. Benerin kata nempel
            1. Buang URL & email
            2. Buang char aneh
            3. Rapiin spasi
        
        Args:
            text (str): Raw input
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Step 0: Fix kata nempel dulu SEBELUM cleaning lain
        # Ini penting untuk PDF yang tidak punya spasi antar kata
        text = self.fix_stuck_words(text)
        
        # Step 0.5: Fix spasi setelah tanda baca
        text = self.fix_punctuation_spacing(text)
        
        # Step 1: Hapus URL dan email
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        
        # Step 2: Hapus special characters (keep punctuation)
        # Sekarang juga normalize Unicode whitespace variants
        text = self.remove_special_chars(text, keep_punctuation=True)
        
        # Step 3: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    # ========================================================================
    # TOKENIZATION FUNCTIONS
    # ========================================================================
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Pecah jadi kalimat.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List kalimat
            
        Example:
            >>> processor = TextPreprocessor()
            >>> sentences = processor.tokenize_sentences("Hello world. How are you?")
            >>> print(sentences)
            ['Hello world.', 'How are you?']
        """
        try:
            sentences = sent_tokenize(text)
            # Filter kalimat kosong
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            print(f"[ERROR] Error dalam sentence tokenization: {e}")
            # Fallback: split by periods
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Pecah jadi kata.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List kata (lowercase)
            
        Example:
            >>> processor = TextPreprocessor()
            >>> words = processor.tokenize_words("Hello World")
            >>> print(words)
            ['hello', 'world']
        """
        try:
            words = word_tokenize(text.lower())
            return words
        except Exception as e:
            print(f"[ERROR] Error dalam word tokenization: {e}")
            # Fallback: simple split
            return text.lower().split()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        Buang kata umum (stopwords).
        
        Args:
            words (List[str]): List kata
            
        Returns:
            List[str]: List kata bersih
        """
        return [w for w in words if w not in self.stop_words and w not in string.punctuation]
    
    # ========================================================================
    # MAIN PREPROCESSING FUNCTION
    # ========================================================================
    
    def preprocess_text(self, text: str, remove_stops=False) -> Dict:
        """
        Jalanin semua proses cleaning & tokenizing.
        
        Args:
            text (str): Raw input text
            remove_stops (bool): Buang stopwords?
            
        Returns:
            dict: Hasil preprocessing (kalimat, kata, stats, dll)
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        # Step 3: Tokenize words
        words = self.tokenize_words(cleaned_text)
        
        # Step 4: (Optional) Remove stopwords
        if remove_stops:
            filtered_words = self.remove_stopwords(words)
        else:
            filtered_words = words
        
        # Buat result dictionary
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'words': words,
            'filtered_words': filtered_words,
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_filtered_words': len(filtered_words)
        }
        
        # Log hasil
        print(f"[OK] Preprocessing selesai:")
        print(f"    Jumlah kalimat: {result['num_sentences']}")
        print(f"    Jumlah kata: {result['num_words']}")
        print(f"    Kata setelah filter: {result['num_filtered_words']}")
        
        return result


# ============================================================================
# CONVENIENCE FUNCTION (untuk kemudahan penggunaan)
# ============================================================================

def preprocess_text(text: str, language='english', remove_stopwords=False) -> Dict:
    """
    Fungsi convenience biar gampang dipake.
    
    Args:
        text (str): Raw input
        language (str): Bahasa (english/indonesian)
        remove_stopwords (bool): Buang stopwords?
        
    Returns:
        dict: Hasil preprocessing
    """
    preprocessor = TextPreprocessor(language=language)
    return preprocessor.preprocess_text(text, remove_stops=remove_stopwords)


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("PREPROCESSING MODULE - TEST")
    print("="*80)
    print()
    
    # Sample text dengan berbagai elemen yang perlu dibersihkan
    sample_text = """
    Artificial Intelligence (AI) has revolutionized technology! Visit https://example.com for more info.
    Machine learning enables systems to learn from data. Contact us at info@example.com.
    Deep learning uses neural networks... Isn't it amazing?!
    """
    
    print(" SAMPLE TEXT:")
    print("-"*80)
    print(sample_text)
    print()
    
    # Test preprocessing
    processor = TextPreprocessor()
    result = processor.preprocess_text(sample_text, remove_stops=True)
    
    print(" CLEANED TEXT:")
    print("-"*80)
    print(result['cleaned_text'])
    print()
    
    print(" SENTENCES:")
    print("-"*80)
    for i, sent in enumerate(result['sentences'], 1):
        print(f"   {i}. {sent}")
    print()
    
    print(" WORDS (first 20 with stopwords):")
    print("-"*80)
    print(result['words'][:20])
    print()
    
    print(" FILTERED WORDS (first 20, no stopwords):")
    print("-"*80)
    print(result['filtered_words'][:20])
    print()
    
    print(" STATISTICS:")
    print("-"*80)
    print(f"   Kalimat: {result['num_sentences']}")
    print(f"   Kata (dengan stopwords): {result['num_words']}")
    print(f"   Kata (tanpa stopwords): {result['num_filtered_words']}")
