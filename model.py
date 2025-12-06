import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from rouge import Rouge
from razdel import sentenize
import numpy as np
import random
from tqdm.auto import tqdm

from google.colab import drive
drive.mount('/content/drive')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

CONFIG = {
    "model_name": "cointegrated/rubert-tiny",
    "data_path_texts": "/content/drive/MyDrive/Summarization_Project/texts",     
    "data_path_summaries": "/content/drive/MyDrive/Summarization_Project/summaries", 
    "max_len": 64,         
    "batch_size": 16,
    "epochs": 4,         
    "learning_rate": 2e-5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "target_rouge": 0.4
}

print(f"Device: {CONFIG['device']}")

def read_file(path):
    """Читает текстовый файл с учетом кодировки."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='cp1251') as f:
            return f.read().strip()

def split_sentences(text):
    """Разбивает текст на предложения с помощью razdel."""
    return [s.text for s in sentenize(text)]

def get_greedy_match(book_sents, summary_sents, threshold=0.55):
    """
    Алгоритм создания "Золотого стандарта" (Oracle summary).
    Для каждого предложения из саммари ищет наиболее похожее предложение в книге
    и помечает его как '1' (важное).
    """
    labels = [0] * len(book_sents)
    rouge = Rouge()

    summary_tokens = [set(s.lower().split()) for s in summary_sents]
    book_tokens = [set(s.lower().split()) for s in book_sents]

    for summ_tok in summary_tokens:
        best_idx = -1
        best_score = 0.0

        if len(summ_tok) < 3: continue

        for idx, book_tok in enumerate(book_tokens):
            if len(book_tok) < 3: continue

            intersection = len(summ_tok.intersection(book_tok))
            union = len(summ_tok.union(book_tok))
            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx != -1 and best_score > 0.2:
            labels[best_idx] = 1

    return labels

def load_dataset_from_drive(texts_dir, summaries_dir):
    """Загружает пары (книга, саммари) из папок."""
    data = []

    text_files = sorted(glob.glob(os.path.join(texts_dir, "*.txt")))

    print(f"Найдено книг: {len(text_files)}")

    for text_path in tqdm(text_files, desc="Чтение и разметка книг"):
        filename = os.path.basename(text_path)
        summary_path = os.path.join(summaries_dir, filename)

        if os.path.exists(summary_path):
            full_text = read_file(text_path)
            summary_text = read_file(summary_path)

            book_sents = split_sentences(full_text)
            summ_sents = split_sentences(summary_text)

            labels = get_greedy_match(book_sents, summ_sents)

            for s, l in zip(book_sents, labels):
                data.append({"text": s, "label": l})
        else:
            print(f"Внимание: Не найдено саммари для {filename}")

    return data

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        label = item['label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

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
        x = self.classifier(x)
        return x

def train_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    model = model.train()
    losses = []

    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.squeeze()

        if logits.dim() == 0: logits = logits.unsqueeze(0)

        loss = loss_fn(logits, targets)

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)

def generate_summary(text, model, tokenizer, top_n=5):
    model.eval()
    sentences = split_sentences(text)
    scores = []

    for sent in sentences:
        inputs = tokenizer.encode_plus(
            sent, return_tensors='pt', max_length=CONFIG['max_len'],
            padding='max_length', truncation=True
        )
        input_ids = inputs['input_ids'].to(CONFIG['device'])
        mask = inputs['attention_mask'].to(CONFIG['device'])

        with torch.no_grad():
            logits = model(input_ids, mask)
            prob = torch.sigmoid(logits).item()
            scores.append((sent, prob))

    top_sentences = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    final_summary_sents = []
    for sent in sentences:
        if any(s[0] == sent for s in top_sentences):
            final_summary_sents.append(sent)

    return " ".join(final_summary_sents)

print("--- Старт загрузки данных ---")
raw_data = load_dataset_from_drive(CONFIG['data_path_texts'], CONFIG['data_path_summaries'])

if len(raw_data) == 0:
    print("ОШИБКА: Данные не найдены. Проверьте пути на Google Drive!")
else:
    print(f"Всего предложений для обучения: {len(raw_data)}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    dataset = SummarizationDataset(raw_data, tokenizer, CONFIG['max_len'])

    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'])

    model = BertExtractiveSummarizer(CONFIG['model_name'])
    model = model.to(CONFIG['device'])

    pos_weight = torch.tensor([5.0]).to(CONFIG['device'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print(f"\n--- Старт обучения на {CONFIG['epochs']} эпох ---")
    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, CONFIG['device'], loss_fn)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {train_loss:.4f}")

        torch.save(model.state_dict(), f"/content/drive/MyDrive/Summarization_Project/bert_summ_ep{epoch+1}.bin")

    print("\nОбучение завершено. Чекпоинты сохранены на Диске.")

    test_files = glob.glob(os.path.join(CONFIG['data_path_texts'], "*.txt"))
    if test_files:
        test_text_path = test_files[0]
        ref_path = os.path.join(CONFIG['data_path_summaries'], os.path.basename(test_text_path))

        full_text = read_file(test_text_path)
        reference = read_file(ref_path)

        generated = generate_summary(full_text, model, tokenizer, top_n=5)

        print("\n=== РЕЗУЛЬТАТ ===")
        print("Сгенерированное саммари:\n", generated[:500], "...") 

        rouge = Rouge()
        try:
            scores = rouge.get_scores(generated, reference)[0]
            print(f"\nROUGE-L F1: {scores['rouge-l']['f']:.4f}")
            if scores['rouge-l']['f'] > CONFIG['target_rouge']:
                print("Цель достигнута!")
            else:
                print(f"Цель {CONFIG['target_rouge']} пока не достигнута.")
        except Exception as e:
            print("Ошибка при подсчете ROUGE (возможно пустой вывод):", e)