# prepare_data.py
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import random
from rouge_score import rouge_scorer
from utils import split_to_chapters, split_to_sentences, textrank_offline_summary
from configs import BOOKS_DIR, OUTPUT_DIR, TOP_K_LABEL, RANDOM_SEED
import pandas as pd

random.seed(RANDOM_SEED)

def prepare(books_dir: Path, out_csv: Path, top_k=TOP_K_LABEL):
    rows = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    books = list(books_dir.glob("*.txt"))
    if not books:
        raise RuntimeError(f"No .txt books found in {books_dir}")
    for p in books:
        text = p.read_text(encoding='utf-8')
        chapters = split_to_chapters(text)
        for ci, ch in enumerate(chapters):
            sents = split_to_sentences(ch)
            if not sents:
                continue
            ref = textrank_offline_summary(ch, ratio=0.2)
            if not ref or len(ref.split()) < 10:
                # fallback
                ref = " ".join(sents[:min(3, len(sents))])
            for si, s in enumerate(sents):
                rows.append({
                    "book": p.name,
                    "chapter_id": ci,
                    "sentence_id": si,
                    "sentence": s,
                    "reference_summary": ref,
                    "chapter_text": ch
                })
    # labeling
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['book'], r['chapter_id'])].append(r)
    labeled = []
    for key, group in grouped.items():
        for r in group:
            r['rougeL'] = scorer.score(r['sentence'], r['reference_summary'])['rougeL'].fmeasure
        group_sorted = sorted(group, key=lambda x: x['rougeL'], reverse=True)
        if len(group_sorted) <= top_k:
            pos_ids = {group_sorted[0]['sentence_id']}
        else:
            pos_ids = set(t['sentence_id'] for t in group_sorted[:top_k])
        for r in group:
            r['label'] = 1 if r['sentence_id'] in pos_ids else 0
            labeled.append(r)
    df = pd.DataFrame(labeled)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved dataset to", out_csv)
    print("Total sentences:", len(df), "Positives:", int(df['label'].sum()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_dir", default=str(BOOKS_DIR))
    parser.add_argument("--out_csv", default=str(OUTPUT_DIR / "dataset.csv"))
    parser.add_argument("--top_k", type=int, default=TOP_K_LABEL)
    args = parser.parse_args()
    prepare(Path(args.books_dir), Path(args.out_csv), top_k=args.top_k)

