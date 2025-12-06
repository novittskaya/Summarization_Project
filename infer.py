# infer.py
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import split_to_chapters, split_to_sentences
from configs import MODEL_DIR, OUTPUT_DIR, TOP_N_SUMMARY
import argparse

def make_summaries_for_book(book_path: Path, model_dir: Path, out_dir: Path, top_n=TOP_N_SUMMARY):
    text = book_path.read_text(encoding='utf-8')
    chapters = split_to_chapters(text)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    all_chap_summaries = []
    for ci, ch in enumerate(chapters):
        sents = split_to_sentences(ch)
        if not sents:
            continue
        enc = tokenizer(sents, truncation=True, padding=True, return_tensors="pt", max_length=256).to(device)
        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[:,1].cpu().numpy()
        import numpy as np
        idxs = np.argsort(-probs)[:min(top_n, len(sents))]
        idxs = sorted(idxs.tolist())
        chosen = [sents[i] for i in idxs]
        chap_summary = " ".join(chosen)
        all_chap_summaries.append(chap_summary)
        out_chap = out_dir / f"{book_path.stem}__chap{ci}.txt"
        out_chap.write_text(chap_summary, encoding='utf-8')
    # full book summary
    full_summary = "\n\n".join(all_chap_summaries)
    out_file = out_dir / f"{book_path.stem}__full_summary.txt"
    out_file.write_text(full_summary, encoding='utf-8')
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", required=True, help="Path to book .txt")
    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    parser.add_argument("--out_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=TOP_N_SUMMARY)
    args = parser.parse_args()
    out = make_summaries_for_book(Path(args.book_path), Path(args.model_dir), Path(args.out_dir), top_n=args.top_n)
    print("Saved summary to", out)

