# utils.py
import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from rutermextract import TermExtractor

nltk.download("punkt")
# punkt_tab not always present in some envs; optional
try:
    nltk.download("punkt_tab")
except:
    pass

extractor = TermExtractor()

def split_to_chapters(text: str) -> List[str]:
    # Простая эвристика: разбиваем по заголовкам "Глава", пустым строкам и крупным блокам
    parts = re.split(r'\n{2,}|\n(?=Глава\s)|\n(?=ЧАСТЬ\s)|\n(?=Часть\s)', text)
    chapters = []
    for p in parts:
        s = p.strip()
        if len(s) > 80:
            chapters.append(s)
    if not chapters:
        # fallback: whole text as single chapter
        chapters = [text]
    return chapters

def split_to_sentences(text: str) -> List[str]:
    sents = sent_tokenize(text, language="russian")
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def textrank_offline_summary(text: str, ratio: float = 0.2, min_sentences: int = 1) -> str:
    sents = split_to_sentences(text)
    if len(sents) <= min_sentences:
        return " ".join(sents)
    # извлекаем термины
    try:
        terms = list(extractor(text))
    except Exception:
        terms = []
    # score each sentence by presence of extracted terms
    scores = []
    norm_terms = [t.normalized for t in terms if hasattr(t, "normalized")]
    for s in sents:
        low = s.lower()
        score = sum(1 for t in norm_terms if t in low)
        scores.append(score)
    # if extractor returns nothing, fallback to simple heuristic: sentence length
    if sum(scores) == 0:
        scores = [len(s) for s in sents]
    # choose top N by ratio
    top_n = max(1, int(len(sents) * ratio))
    idxs = sorted(range(len(sents)), key=lambda i: -scores[i])[:top_n]
    idxs = sorted(idxs)
    selected = [sents[i] for i in idxs]
    return " ".join(selected)

