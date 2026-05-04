import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEAK_LABELS_FILE = (
    PROJECT_ROOT / "LF" / "improvements_aayush" / "results" / "Weak_Labels_10k" / "Snorkel_Weak_Labels_10k.csv"
)
GOLD_FILE = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step4_Snorkel_Results"
    / "Step4_Snorkel_Results_portable.xlsx"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Stage3_Weak_Pretrain"
    / "Stage3_Weak_Pretrain_Results.xlsx"
)
MODEL_OUTPUT_DIR = (
    PROJECT_ROOT / "LF" / "improvements_aayush" / "results" / "bert_model_weak_pretrained"
)

LABEL2ID = {"URGENT": 0, "ACTION": 1, "INFORMATION": 2}
ID2LABEL = {0: "URGENT", 1: "ACTION", 2: "INFORMATION"}
CLASSES = ["URGENT", "ACTION", "INFORMATION"]


def clean_body(body):
    body = re.sub(r"-----original message-----.*", "", str(body), flags=re.DOTALL)
    body = re.sub(r"---------------------- forwarded by.*", "", body, flags=re.DOTALL)
    body = re.sub(r"on .{5,50} wrote:.*", "", body, flags=re.DOTALL)
    return body.strip()


def make_text(row):
    subj = str(row.get("subject", "") or row.get("Subject", "") or "")
    body = clean_body(row.get("body", "") or row.get("Body", "") or "")
    return f"{subj} [SEP] {body}"


def load_datasets():
    weak_df = pd.read_csv(WEAK_LABELS_FILE)
    weak_df["text"] = weak_df.apply(make_text, axis=1)
    weak_df = weak_df.dropna(subset=["P_URGENT", "P_ACTION", "P_INFORMATION"]).copy()

    gold_df = pd.read_excel(GOLD_FILE, sheet_name="Snorkel_Results")
    gold_df["text"] = gold_df.apply(make_text, axis=1)
    gold_df["gold"] = gold_df["Final Label"].str.strip().str.upper()
    gold_df = gold_df[gold_df["gold"] != "TIE"].copy()
    gold_df = gold_df[gold_df["gold"].isin(CLASSES)].copy()
    return weak_df.reset_index(drop=True), gold_df.reset_index(drop=True)


def evaluate(y_true, y_pred, model_name):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
    for cls in CLASSES:
        corr = sum((np.array(y_true) == cls) & (np.array(y_pred) == cls))
        tot_c = sum(np.array(y_true) == cls)
        print(f"  {cls:<14}: {corr}/{tot_c} ({corr / tot_c * 100 if tot_c else 0:.1f}%)")
    print()
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':14} {'URGENT':>8} {'ACTION':>8} {'INFO':>8}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<14} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    return {"accuracy": correct / total, "confusion_matrix": cm}


def run_pipeline():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch/Transformers are required for Stage 3.") from exc

    weak_df, gold_df = load_datasets()
    print(f"Weak-label rows: {len(weak_df)}")
    print(f"Gold rows: {len(gold_df)}")

    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 8
    PRETRAIN_EPOCHS = 1
    FINETUNE_EPOCHS = 4
    LR = 1e-5
    GRAD_ACCUM = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class SoftLabelDataset(Dataset):
        def __init__(self, texts, probs, tokenizer, max_len):
            self.texts = texts
            self.probs = probs
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "soft_label": torch.tensor(self.probs[idx], dtype=torch.float),
            }

    class HardLabelDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    def train_weak_then_gold(model, weak_dl, train_dl, weights):
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        model.train()
        for epoch in range(PRETRAIN_EPOCHS):
            optimizer.zero_grad()
            total_loss = 0.0
            for step, batch in enumerate(weak_dl):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                soft_label = batch["soft_label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)
                loss = kl_loss(log_probs, soft_label) / GRAD_ACCUM
                loss.backward()
                total_loss += loss.item() * GRAD_ACCUM
                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(weak_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            print(f"  Pretrain epoch {epoch + 1}/{PRETRAIN_EPOCHS} — loss: {total_loss / len(weak_dl):.4f}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        ce_loss = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for epoch in range(FINETUNE_EPOCHS):
            optimizer.zero_grad()
            total_loss = 0.0
            for step, batch in enumerate(train_dl):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = ce_loss(outputs.logits, labels) / GRAD_ACCUM
                loss.backward()
                total_loss += loss.item() * GRAD_ACCUM
                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            print(f"  Fine-tune epoch {epoch + 1}/{FINETUNE_EPOCHS} — loss: {total_loss / len(train_dl):.4f}")

    weak_texts = weak_df["text"].tolist()
    weak_probs = weak_df[["P_URGENT", "P_ACTION", "P_INFORMATION"]].to_numpy(dtype=np.float32)
    gold_texts = gold_df["text"].tolist()
    gold_labels = np.array([LABEL2ID[g] for g in gold_df["gold"]])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = [None] * len(gold_df)
    all_probs = [None] * len(gold_df)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.array(gold_texts), gold_labels), 1):
        print(f"\nFold {fold}/5")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(device)

        weak_ds = SoftLabelDataset(weak_texts, weak_probs, tokenizer, MAX_LEN)
        weak_dl = DataLoader(weak_ds, batch_size=BATCH_SIZE, shuffle=True)
        train_texts = [gold_texts[i] for i in train_idx]
        train_labels = [gold_labels[i] for i in train_idx]
        val_texts = [gold_texts[i] for i in val_idx]
        val_labels = [gold_labels[i] for i in val_idx]

        from collections import Counter

        counts = Counter(train_labels)
        n = len(train_labels)
        raw_weights = [n / (3 * counts[i]) for i in range(3)]
        capped_weights = [min(w, 3.0) for w in raw_weights]
        weights = torch.tensor(capped_weights, dtype=torch.float).to(device)

        train_ds = HardLabelDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_ds = HardLabelDataset(val_texts, val_labels, tokenizer, MAX_LEN)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        train_weak_then_gold(model, weak_dl, train_dl, weights)

        model.eval()
        with torch.no_grad():
            val_offset = 0
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)
                batch_val_idx = val_idx[val_offset : val_offset + len(preds)]
                for i, global_idx in enumerate(batch_val_idx):
                    all_preds[global_idx] = ID2LABEL[preds[i]]
                    all_probs[global_idx] = probs[i]
                val_offset += len(preds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_true = [ID2LABEL[l] for l in gold_labels]
    results = evaluate(y_true, all_preds, "BERT (weak pretraining + gold fine-tuning, 5-fold CV)")

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    out = gold_df[["subject", "body", "gold"]].copy()
    out.columns = ["Subject", "Body", "Gold Label"]
    out["Stage3_BERT_Prediction"] = all_preds
    out["Stage3_BERT_Correct"] = (
        out["Gold Label"].astype(str).str.strip().str.upper()
        == out["Stage3_BERT_Prediction"].astype(str).str.strip().str.upper()
    ).map({True: "✓", False: "✗"})
    out["Stage3_P_URGENT"] = np.array(all_probs)[:, 0]
    out["Stage3_P_ACTION"] = np.array(all_probs)[:, 1]
    out["Stage3_P_INFORMATION"] = np.array(all_probs)[:, 2]

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="Results")
        pd.DataFrame(
            [{"Model": "Weak pretraining + gold fine-tuning BERT", "Accuracy %": round(results["accuracy"] * 100, 2)}]
        ).to_excel(writer, index=False, sheet_name="Summary")
    print(f"\nSaved Stage 3 results → {OUTPUT_FILE}")

    print("\nTraining final Stage 3 model on all data for saving...")
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    weak_ds = SoftLabelDataset(weak_texts, weak_probs, tokenizer, MAX_LEN)
    weak_dl = DataLoader(weak_ds, batch_size=BATCH_SIZE, shuffle=True)

    from collections import Counter

    final_counts = Counter(gold_labels)
    final_n = len(gold_labels)
    final_raw_weights = [final_n / (3 * final_counts[i]) for i in range(3)]
    final_capped_weights = [min(w, 3.0) for w in final_raw_weights]
    final_weights = torch.tensor(final_capped_weights, dtype=torch.float).to(device)

    final_gold_ds = HardLabelDataset(gold_texts, gold_labels.tolist(), tokenizer, MAX_LEN)
    final_gold_dl = DataLoader(final_gold_ds, batch_size=BATCH_SIZE, shuffle=True)

    train_weak_then_gold(final_model, weak_dl, final_gold_dl, final_weights)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    final_model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Saved final Stage 3 model → {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()
