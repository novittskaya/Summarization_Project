import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from razdel import sentenize
import math

FILENAME = "test.txt" 

# Пути
BASE_PATH = "/content/drive/MyDrive/Summarization_Project"
MODEL_PATH = os.path.join(BASE_PATH, "bert_summ_ep4.bin")
TEXTS_PATH = os.path.join(BASE_PATH)
MODEL_NAME = "cointegrated/rubert-tiny"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertExtractiveSummarizer(nn.Module):
    def __init__(self, model_name):
        super(BertExtractiveSummarizer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.classifier(x)

print("⏳ Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BertExtractiveSummarizer(MODEL_NAME)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Модель успешно загружена из файла!")
else:
    raise FileNotFoundError("Веса модели не найдены!")


def rate_sentences(sentences, model, tokenizer):
    """Присваивает каждому предложению оценку важности (0..1)."""
    scores = []
    for i, sent in enumerate(sentences):
        if len(sent) < 15: 
            scores.append((sent, 0.0, i))
            continue
            
        try:
            inputs = tokenizer.encode_plus(
                sent, return_tensors='pt', max_length=128, 
                padding='max_length', truncation=True
            )
            input_ids = inputs['input_ids'].to(DEVICE)
            mask = inputs['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_ids, mask)
                score = torch.sigmoid(logits).item()
                scores.append((sent, score, i))
        except:
             scores.append((sent, 0.0, i))
    return scores

def summarize_adaptive(filename):
    input_file = os.path.join(TEXTS_PATH, filename)
    if not os.path.exists(input_file):
        print("Файл не найден!")
        return

    print(f"Читаю {filename}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ')

    all_sentences = [s.text for s in sentenize(text)]
    total_sents = len(all_sentences)
    print(f"Всего предложений: {total_sents}")

    final_summary_sentences = []
    
    scores = rate_sentences(all_sentences, model, tokenizer)

    if total_sents < 100:
        print("⚡ Режим: Short Story (Buckets + Smoothing)")
        
        target_count = max(3, int(total_sents * 0.25))
        bucket_size = math.ceil(total_sents / target_count)
        
        selected_indices = set()
        
        print("   -> Добавлено вступление (sentence #0)")
        selected_indices.add(0)

        for i in range(target_count):
            start_idx = i * bucket_size
            end_idx = min((i + 1) * bucket_size, total_sents)
            if start_idx >= total_sents: break
            
            sector_scores = scores[start_idx:end_idx]
            valid_candidates = [s for s in sector_scores if s[1] > 0]
            
            if valid_candidates:
                best_in_sector = max(valid_candidates, key=lambda x: x[1])
                selected_indices.add(best_in_sector[2])

        current_indices = sorted(list(selected_indices))
        for idx in current_indices:
            if idx > 0:
                prev_idx = idx - 1
                if prev_idx not in selected_indices and len(all_sentences[prev_idx]) > 20:
                    first_word = all_sentences[idx].split()[0].lower()
                    if first_word in ['она', 'он', 'мы', 'они', 'я']:
                        print(f"   -> Контекст (местоимение): к №{idx} добавлено №{prev_idx}")
                        selected_indices.add(prev_idx)

        sorted_indices = sorted(list(selected_indices))
        final_indices = set(selected_indices)
        
        for i in range(len(sorted_indices) - 1):
            current = sorted_indices[i]
            next_val = sorted_indices[i+1]
            gap = next_val - current - 1
            
            if 0 < gap <= 2:
                for fill_idx in range(current + 1, next_val):
                    print(f"   -> Заделка шва: добавлено пропущенное №{fill_idx}")
                    final_indices.add(fill_idx)

        final_summary_sentences = [all_sentences[i] for i in sorted(list(final_indices))]

    else:
        print("Режим: Книга")
        CHUNK_SIZE = 50       
        TOP_N_PER_CHUNK = 2   
        for i in range(0, total_sents, CHUNK_SIZE):
            chunk_scores = scores[i : i + CHUNK_SIZE]
            top_items = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:TOP_N_PER_CHUNK]
            final_summary_sentences.extend([x[0] for x in top_items])

    result_text = " ".join(final_summary_sentences)
    
    output_filename = f"SMART_SUMMARY_{filename}"
    save_path = os.path.join(BASE_PATH, output_filename)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Источник: {filename}\n")
        f.write(f"Режим: {'Short+Glue' if total_sents < 100 else 'Novel'}\n")
        f.write("-" * 20 + "\n")
        f.write(result_text)
        
    print(f"Готово! Результат в файле: {output_filename}")
    print("\n--- ПРЕДПРОСМОТР РЕЗУЛЬТАТА ---")
    print(result_text[:1000])

summarize_adaptive(FILENAME)