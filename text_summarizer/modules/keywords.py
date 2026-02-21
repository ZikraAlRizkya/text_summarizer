# ============================================================
# KEYWORD EXTRACTION MODULE
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    keywords = [word for word, score in sorted_scores[:top_k]]
    return keywords