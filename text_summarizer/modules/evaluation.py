from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer


def calculate_rouge(reference: str, candidate: str) -> Dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


def cosine_similarity_score(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return float(score)


def compression_ratio(original: str, summary: str) -> float:
    if len(original) == 0:
        return 0.0
    return len(summary) / len(original)


def evaluate_all(original: str, extractive: str, abstractive: str) -> Dict:
    return {
        "extractive": {
            "rouge": calculate_rouge(original, extractive),
            "similarity": cosine_similarity_score(original, extractive),
            "compression": compression_ratio(original, extractive)
        },
        "abstractive": {
            "rouge": calculate_rouge(original, abstractive),
            "similarity": cosine_similarity_score(original, abstractive),
            "compression": compression_ratio(original, abstractive)
        }
    }