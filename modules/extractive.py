"""
================================================================================
EXTRACTIVE SUMMARIZATION MODULE
================================================================================
Implementasi algoritma TextRank untuk extractive summarization

Author: Member 1 - NLP & Backend Engineer
Phase: 2 (Core Summarization)

Algorithm: TextRank (PageRank untuk teks)
    1. Hitung similarity matrix antar kalimat (TF-IDF + Cosine Similarity)
    2. Buat graph dari similarity matrix
    3. Apply PageRank algorithm untuk ranking kalimat
    4. Pilih top-N kalimat dengan ranking tertinggi
    5. Return summary dalam urutan asli

Classes:
    - TextRankSummarizer: Class utama untuk extractive summarization

Usage:
    from modules.extractive import TextRankSummarizer
    
    summarizer = TextRankSummarizer()
    result = summarizer.summarize(text=text, ratio=0.3)
    print(result['summary'])
================================================================================
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    """
    TextRank-based extractive summarization
    
    TextRank menggunakan graph-based ranking algorithm (mirip Google PageRank)
    untuk menentukan kalimat mana yang paling penting dalam sebuah teks.
    """
    
    def __init__(self, similarity_threshold=0.1):
        """
        Inisialisasi TextRank summarizer
        
        Args:
            similarity_threshold (float): Threshold minimum untuk membuat edge
                                         dalam graph (default: 0.1)
        """
        self.similarity_threshold = similarity_threshold
    
    # ========================================================================
    # STEP 1: COMPUTE SIMILARITY MATRIX
    # ========================================================================
    
    def compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Menghitung similarity matrix antar kalimat menggunakan TF-IDF dan
        cosine similarity
        
        Penjelasan:
            - TF-IDF: Representasi kalimat dalam bentuk vektor berdasarkan
                     frekuensi kata (Term Frequency - Inverse Document Frequency)
            - Cosine Similarity: Mengukur seberapa mirip dua kalimat
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            np.ndarray: Similarity matrix (N x N) dimana N = jumlah kalimat
                       matrix[i][j] = similarity antara kalimat i dan j
        """
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            # Buat TF-IDF matrix
            # TF-IDF mengubah kalimat menjadi vektor numerik
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Hitung cosine similarity
            # Cosine similarity mengukur sudut antara dua vektor
            # Nilai 1 = sangat mirip, nilai 0 = tidak mirip
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"[ERROR] Error saat menghitung similarity matrix: {e}")
            # Fallback: return identity matrix (diagonal = 1, sisanya = 0)
            n = len(sentences)
            return np.eye(n)
    
    # ========================================================================
    # STEP 2: BUILD GRAPH
    # ========================================================================
    
    def build_similarity_graph(self, similarity_matrix: np.ndarray) -> nx.Graph:
        """
        Membuat graph dari similarity matrix
        
        Penjelasan:
            - Node: Setiap kalimat adalah node
            - Edge: Dua kalimat terhubung jika similarity > threshold
            - Weight: Bobot edge = nilai similarity
        
        Args:
            similarity_matrix (np.ndarray): Matrix similarity
            
        Returns:
            nx.Graph: Graph dengan weighted edges
        """
        n = similarity_matrix.shape[0]
        graph = nx.Graph()
        
        # Tambahkan nodes (kalimat)
        graph.add_nodes_from(range(n))
        
        # Tambahkan edges dengan weights
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                # Hanya tambahkan edge jika similarity > threshold
                if similarity > self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
        
        return graph
    
    # ========================================================================
    # STEP 3: RANK SENTENCES
    # ========================================================================
    
    def rank_sentences(self, graph: nx.Graph) -> Dict[int, float]:
        """
        Ranking kalimat menggunakan PageRank algorithm
        
        Penjelasan PageRank:
            - Algoritma yang digunakan Google untuk ranking web pages
            - Idea: Halaman penting jika banyak halaman penting yang link ke dia
            - Dalam konteks teks: Kalimat penting jika mirip dengan banyak
              kalimat penting lainnya
        
        Args:
            graph (nx.Graph): Similarity graph
            
        Returns:
            dict: Dictionary {index_kalimat: score}
        """
        try:
            # Apply PageRank algorithm
            scores = nx.pagerank(graph, weight='weight')
            return scores
        except:
            # Fallback: uniform scores jika PageRank gagal
            print("[WARNING]  Warning: PageRank failed, menggunakan uniform scores")
            return {i: 1.0 for i in graph.nodes()}
    
    # ========================================================================
    # STEP 4: SELECT TOP SENTENCES
    # ========================================================================
    
    def select_top_sentences(
        self, 
        sentences: List[str], 
        scores: Dict[int, float], 
        num_sentences: Optional[int] = None,
        ratio: float = 0.3,
        ensure_coverage: bool = True
    ) -> List[Tuple[int, str, float, str]]:
        """
        Memilih top-ranked sentences dengan coverage-aware selection
        
        Args:
            sentences (List[str]): Original sentences
            scores (dict): Sentence scores dari PageRank
            num_sentences (int): Jumlah kalimat yang diinginkan
                                (jika None, gunakan ratio)
            ratio (float): Rasio kalimat yang dipilih (default: 30%)
            ensure_coverage (bool): Pastikan semua section terwakili (default: True)
            
        Returns:
            List[Tuple[int, str, float, str]]: List of (index, sentence, score, position)
                                               diurutkan berdasarkan posisi asli
        """
        # Tentukan jumlah kalimat yang akan dipilih
        if num_sentences is None:
            num_sentences = max(1, int(len(sentences) * ratio))
        
        # Pastikan tidak melebihi jumlah kalimat yang ada
        num_sentences = min(num_sentences, len(sentences))
        
        total_sentences = len(sentences)
        
        if ensure_coverage and num_sentences >= 3 and total_sentences >= 3:
            # COVERAGE-AWARE SELECTION
            # Bagi dokumen jadi 3 section: Awal, Tengah, Akhir
            section_size = total_sentences / 3
            sections = {
                'Awal': list(range(0, int(section_size))),
                'Tengah': list(range(int(section_size), int(2 * section_size))),
                'Akhir': list(range(int(2 * section_size), total_sentences))
            }
            
            # Alokasi minimal 1 kalimat per section
            selected = []
            remaining_quota = num_sentences
            
            # Untuk setiap section, ambil kalimat dengan score tertinggi
            for section_name, indices in sections.items():
                if remaining_quota <= 0:
                    break
                    
                # Cari kalimat terbaik di section ini yang belum dipilih
                section_scores = [(idx, scores.get(idx, 0.0)) for idx in indices]
                section_scores.sort(key=lambda x: x[1], reverse=True)
                
                if section_scores:
                    best_idx = section_scores[0][0]
                    selected.append((best_idx, sentences[best_idx], scores[best_idx], section_name))
                    remaining_quota -= 1
            
            # Sisa quota: ambil kalimat dengan score tertinggi yang belum dipilih
            selected_indices = {s[0] for s in selected}
            remaining_candidates = [
                (idx, sentences[idx], scores[idx])
                for idx in range(total_sentences)
                if idx not in selected_indices
            ]
            remaining_candidates.sort(key=lambda x: x[2], reverse=True)
            
            for idx, sent, score in remaining_candidates[:remaining_quota]:
                # Tentukan position
                position = self._get_position(idx, total_sentences)
                selected.append((idx, sent, score, position))
            
            # Sort by original position
            selected.sort(key=lambda x: x[0])
            top_sentences = selected
            
        else:
            # STANDARD SELECTION (untuk dokumen kecil atau num_sentences < 3)
            ranked_sentences = [
                (idx, sentences[idx], scores[idx]) 
                for idx in range(len(sentences))
            ]
            ranked_sentences.sort(key=lambda x: x[2], reverse=True)
            
            # Ambil top N kalimat
            top_sentences = ranked_sentences[:num_sentences]
            
            # Tambahkan position info
            top_sentences = [
                (idx, sent, score, self._get_position(idx, total_sentences))
                for idx, sent, score in top_sentences
            ]
            
            # Sort by original position
            top_sentences.sort(key=lambda x: x[0])
        
        return top_sentences
    
    def _get_position(self, idx: int, total: int) -> str:
        """
        Helper function untuk determine position category
        
        Args:
            idx (int): Sentence index
            total (int): Total sentences
            
        Returns:
            str: 'Awal', 'Tengah', or 'Akhir'
        """
        if idx < total / 3:
            return 'Awal'
        elif idx < 2 * total / 3:
            return 'Tengah'
        else:
            return 'Akhir'
    
    # ========================================================================
    # MAIN SUMMARIZATION FUNCTION
    # ========================================================================
    
    def summarize(
        self, 
        text: Optional[str] = None,
        sentences: Optional[List[str]] = None,
        num_sentences: Optional[int] = None,
        ratio: float = 0.3
    ) -> Dict:
        """
        Generate extractive summary menggunakan TextRank
        
        Args:
            text (str): Input text (jika sentences belum di-tokenize)
            sentences (List[str]): Pre-tokenized sentences
            num_sentences (int): Jumlah kalimat dalam summary
            ratio (float): Rasio summary (default: 30%)
            
        Returns:
            dict: Dictionary berisi:
                - summary: String summary
                - sentences: List of (index, sentence, score, position)
                - scores: Dictionary semua sentence scores
                - num_sentences: Jumlah kalimat dalam summary
                - compression_ratio: Rasio kompresi
                
        Example:
            >>> summarizer = TextRankSummarizer()
            >>> result = summarizer.summarize(text="Long text...", ratio=0.3)
            >>> print(result['summary'])
            >>> print(f"Compression: {result['compression_ratio']:.2%}")
        """
        # Import preprocessing di sini untuk avoid circular dependency
        from .preprocessing import TextPreprocessor
        
        # Dapatkan kalimat
        if sentences is None:
            if text is None:
                raise ValueError("Harus menyediakan text atau sentences")
            preprocessor = TextPreprocessor()
            sentences = preprocessor.tokenize_sentences(text)
        
        # Validasi: cek apakah ada kalimat
        if len(sentences) == 0:
            return {
                'summary': "",
                'sentences': [],
                'scores': {},
                'num_sentences': 0
            }
        
        print(f" Extractive Summarization (TextRank):")
        print(f"   Total kalimat input: {len(sentences)}")
        
        # STEP 1: Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(sentences)
        print(f"   [OK] Similarity matrix dihitung")
        
        # STEP 2: Build graph
        graph = self.build_similarity_graph(similarity_matrix)
        print(f"   [OK] Graph dibuat: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")
        
        # STEP 3: Rank sentences
        scores = self.rank_sentences(graph)
        print(f"   [OK] Kalimat di-rank menggunakan PageRank")
        
        # STEP 4: Select top sentences (with coverage-aware selection)
        top_sentences = self.select_top_sentences(
            sentences, scores, num_sentences, ratio
        )
        print(f"   [OK] Dipilih {len(top_sentences)} kalimat:")
        
        # Display selected sentences with position info
        for idx, sent, score, position in top_sentences:
            print(f"      - Kalimat #{idx+1} [{position}] (score: {score:.4f})")
        
        # Generate summary text (extract just the sentence text)
        summary = " ".join([sent for idx, sent, score, position in top_sentences])
        
        # Buat result dictionary
        result = {
            'summary': summary,
            'sentences': top_sentences,  # Now includes position
            'scores': scores,
            'num_sentences': len(top_sentences),
            'compression_ratio': len(top_sentences) / len(sentences)
        }
        
        return result


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def extractive_summary(
    text: Optional[str] = None,
    sentences: Optional[List[str]] = None,
    num_sentences: Optional[int] = None,
    ratio: float = 0.3
) -> str:
    """
    Fungsi convenience untuk generate extractive summary
    
    Args:
        text (str): Input text
        sentences (List[str]): Pre-tokenized sentences
        num_sentences (int): Jumlah kalimat
        ratio (float): Rasio summary
        
    Returns:
        str: Summary text
    """
    summarizer = TextRankSummarizer()
    result = summarizer.summarize(text, sentences, num_sentences, ratio)
    return result['summary']


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EXTRACTIVE SUMMARIZATION (TextRank) - TEST")
    print("="*80)
    print()
    
    # Sample text tentang AI
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
    
    # Test summarization
    summarizer = TextRankSummarizer()
    result = summarizer.summarize(text=sample_text, ratio=0.4)
    
    print("\n EXTRACTIVE SUMMARY (40% dari original):")
    print("-"*80)
    print(result['summary'])
    print()
    
    print(" STATISTICS:")
    print("-"*80)
    print(f"   Compression ratio: {result['compression_ratio']:.2%}")
    print(f"   Kalimat dipilih: {result['num_sentences']}")
    print()
    
    print(" TOP SENTENCES (dengan scores):")
    print("-"*80)
    for i, (idx, sent, score) in enumerate(result['sentences'], 1):
        print(f"   {i}. [Score: {score:.4f}] {sent[:60]}...")
