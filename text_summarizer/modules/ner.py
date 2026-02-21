# ============================================================
# NAMED ENTITY RECOGNITION MODULE
# ============================================================

import spacy
from typing import List, Dict

# Load model (download dulu: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> List[Dict]:
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    
    return entities