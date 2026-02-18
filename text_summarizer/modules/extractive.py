# Modul extractive summarization — milih kalimat paling penting dari teks asli
# pakai algoritma TextRank (versi PageRank buat teks)

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextRankSummarizer:
    # Class utama untuk extractive summarization pakai TextRank

    def __init__(self, similarity_threshold=0.1):
        # threshold: batas minimum similarity supaya dua kalimat dianggap terhubung di graph
        self.similarity_threshold = similarity_threshold
    
    def compute_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        # Ubah tiap kalimat jadi vektor TF-IDF, lalu hitung cosine similarity antar kalimat
        # Hasilnya matrix N x N, nilai[i][j] = seberapa mirip kalimat i dan j
        if len(sentences) < 2:
            return np.array([[1.0]])
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return similarity_matrix
            
        except Exception as e:
            print(f"[ERROR] Error saat menghitung similarity matrix: {e}")
            n = len(sentences)
            return np.eye(n)
    
    def build_similarity_graph(self, similarity_matrix: np.ndarray) -> nx.Graph:
        # Bikin graph: tiap kalimat jadi node, dua kalimat dihubungkan edge
        # kalau similarity-nya di atas threshold
        n = similarity_matrix.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity)
        
        return graph
    
    def rank_sentences(self, graph: nx.Graph) -> Dict[int, float]:
        # Jalanin PageRank di graph — kalimat yang banyak terhubung ke kalimat penting
        # dapat score tinggi (sama kayak cara Google ranking halaman web)
        try:
            scores = nx.pagerank(graph, weight='weight')
            return scores
        except:
            print("[WARNING]  Warning: PageRank failed, menggunakan uniform scores")
            return {i: 1.0 for i in graph.nodes()}
    
    def select_top_sentences(
        self, 
        sentences: List[str], 
        scores: Dict[int, float], 
        num_sentences: Optional[int] = None,
        ratio: float = 0.4,
        ensure_coverage: bool = True,
        include_first_sentence: bool = True
    ) -> List[Tuple[int, str, float, str]]:
        # Pilih top-N kalimat berdasarkan score PageRank
        # ensure_coverage = True: pastikan ada perwakilan dari awal, tengah, akhir teks
        # include_first_sentence = True: force ambil kalimat pertama (bagus buat berita/artikel)
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
                    # Logic khusus buat section 'Awal': force ambil kalimat pertama kalau diminta
                    if include_first_sentence and section_name == 'Awal' and 0 in indices:
                        # Prioritaskan kalimat pertama jika ada di section Awal
                        current_best_idx = section_scores[0][0]
                        if current_best_idx != 0:
                             # Cek score relative, kalau beda jauh banget mungkin keep best?
                             # Tapi user minta flow, jadi prioritaskan 0.
                             best_idx = 0
                        else:
                             best_idx = 0
                    else:
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
            
            # Kalau include_first_sentence=True dan kalimat pertama belum masuk
            if include_first_sentence and num_sentences > 0:
                has_first = any(s[0] == 0 for s in top_sentences)
                if not has_first:
                    # Cari kalimat pertama
                    first_sentence_tuple = next((s for s in ranked_sentences if s[0] == 0), None)
                    if first_sentence_tuple:
                        # Remove yang paling bontot (terakhir di list hasil sort score)
                        # Tapi cek dulu biar ga remove yang penting banget
                        if len(top_sentences) > 0:
                            top_sentences.pop()
                        # Masukin kalimat pertama
                        top_sentences.append(first_sentence_tuple)
            
        # REFERENTIAL CONTINUITY CHECK
        # Cek kalo ada kalimat yang depannya "Hal ini", "Ini", "Itu" dll tapi kalimat sebelumnya ga ada
        # List kata rujukan yang butuh konteks sebelumnya
        ref_keywords = ["hal ini", "hal tersebut", "ini ", "itu ", "karena itu", "oleh karena itu", "maka ", "akibatnya", "namun,"]
        
        # Bikin set index yang udah kepilih biar gampang ngecek
        selected_indices = {s[0] for s in top_sentences}
        to_add = []
        
        # Reconstruct ranked_sentences for iteration (since it wasn't defined in this block)
        ranked_sentences = [
            (idx, sentences[idx], scores.get(idx, 0.0)) 
            for idx in range(total_sentences)
        ]
        
        for idx, sent, score in ranked_sentences: # Iterate all sentences to find the tuple data
             if idx in selected_indices:
                 # Cek apakah kalimat ini butuh pendamping
                 lower_sent = sent.lower().strip()
                 
                 needs_context = any(lower_sent.startswith(k) for k in ref_keywords)
                 
                 if needs_context:
                     prev_idx = idx - 1
                     if prev_idx >= 0 and prev_idx not in selected_indices and prev_idx not in [x[0] for x in to_add]:
                         # Cari tuple buat prev_idx
                         prev_tuple = next((s for s in ranked_sentences if s[0] == prev_idx), None)
                         if prev_tuple:
                             # Masukkan sebagai tambahan (ga usah remove yang lain biar aman)
                             to_add.append(prev_tuple)
                             
        # Gabungin hasil
        top_sentences.extend(to_add)
        
        # Tambahkan position info (karena logic atas mungkin nambah tuple tanpa posisi)
        # Tuple di ranked_sentences isinya (idx, sent, score). top_sentences butuh (idx, sent, score, pos)
        
        # Re-format all top_sentences to ensure they have position
        final_top_sentences = []
        for item in top_sentences:
            # Item bisa berupa length 3 (from ranked) atau length 4 (from coverage selection)
            idx, sent, score = item[0], item[1], item[2]
            pos = self._get_position(idx, total_sentences)
            final_top_sentences.append((idx, sent, score, pos))
            
        top_sentences = final_top_sentences
        
        # Sort by original position (index)
        top_sentences.sort(key=lambda x: x[0])
        
        # Remove duplicates just in case
        unique_top = []
        seen_indices = set()
        for item in top_sentences:
            if item[0] not in seen_indices:
                unique_top.append(item)
                seen_indices.add(item[0])
        top_sentences = unique_top
        
        return top_sentences
    
    def _get_position(self, idx: int, total: int) -> str:
        # Helper: kasih label posisi kalimat (Awal/Tengah/Akhir) berdasarkan indeksnya
        if idx < total / 3:
            return 'Awal'
        elif idx < 2 * total / 3:
            return 'Tengah'
        else:
            return 'Akhir'
    
    def summarize(
        self, 
        text: Optional[str] = None,
        sentences: Optional[List[str]] = None,
        num_sentences: Optional[int] = None,
        ratio: float = 0.4
    ) -> Dict:
        # Fungsi utama: jalanin pipeline TextRank dari awal sampai akhir
        # bisa terima raw text atau list kalimat yang udah di-tokenize
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
            sentences, scores, num_sentences, ratio,
            include_first_sentence=True
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


# Shortcut function — langsung return string summary tanpa perlu bikin object dulu
def extractive_summary(
    text: Optional[str] = None,
    sentences: Optional[List[str]] = None,
    num_sentences: Optional[int] = None,
    ratio: float = 0.4
) -> str:
    summarizer = TextRankSummarizer()
    result = summarizer.summarize(text, sentences, num_sentences, ratio)
    return result['summary']


# Blok testing — jalan kalau file ini dirun langsung (bukan di-import)
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
