"""
STEP 5B: SYNTHETIC URGENT AUGMENTATION EXPERIMENT
=================================================
Goal:
  Add a small synthetic URGENT-only training set to address class imbalance,
  while keeping evaluation strictly on the original gold-labelled emails.

Method:
  - Load the same 294 gold-labelled emails used in Step 5
  - Load a separate synthetic URGENT dataset
  - For each CV fold, append synthetic URGENT emails to the TRAIN split only
  - Evaluate only on the untouched real validation fold

This keeps the comparison honest while testing whether targeted augmentation
helps the rare URGENT class.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNORKEL_OUTPUT = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step4_Snorkel_Results"
    / "Step4_Snorkel_Results_portable.xlsx"
)
BASELINE_RESULTS_FILE = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step5_BERT"
    / "Step5_BERT_Results_portable.xlsx"
)
AUGMENTED_RESULTS_FILE = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step5_BERT"
    / "Step5_Urgent_Augmentation_Results.xlsx"
)
SYNTHETIC_URGENT_FILE = PROJECT_ROOT / "dataset" / "synthetic_urgent_emails.jsonl"
AUGMENTED_BERT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "bert_model_augmented_urgent"
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


def load_gold_data():
    df = pd.read_excel(SNORKEL_OUTPUT, sheet_name="Snorkel_Results")
    df["text"] = df.apply(make_text, axis=1)
    df["gold"] = df["Final Label"].str.strip().str.upper()
    df = df[df["gold"] != "TIE"].copy()
    df = df[df["gold"].isin(CLASSES)].copy()
    print(f"Loaded {len(df)} gold emails")
    print(df["gold"].value_counts().to_string())
    return df.reset_index(drop=True)


def load_synthetic_urgent():
    rows = []
    with open(SYNTHETIC_URGENT_FILE, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["gold"] = df["Final Label"].str.strip().str.upper()
    df["text"] = df.apply(make_text, axis=1)
    print(f"\nLoaded {len(df)} synthetic URGENT emails")
    return df.reset_index(drop=True)


def evaluate(y_true, y_pred, model_name):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
    per_class_rows = []
    for cls in CLASSES:
        corr = sum((np.array(y_true) == cls) & (np.array(y_pred) == cls))
        tot_c = sum(np.array(y_true) == cls)
        pct = corr / tot_c * 100 if tot_c else 0.0
        per_class_rows.append({"Class": cls, "Correct": corr, "Total": tot_c, "Accuracy %": round(pct, 2)})
        print(f"  {cls:<14}: {corr}/{tot_c} ({pct:.1f}%)")
    print()
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':14} {'URGENT':>8} {'ACTION':>8} {'INFO':>8}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<14} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    return {
        "accuracy": correct / total,
        "predictions": list(y_pred),
        "per_class": pd.DataFrame(per_class_rows),
        "confusion_matrix": pd.DataFrame(cm, index=CLASSES, columns=CLASSES),
    }


def load_baseline_predictions():
    if not BASELINE_RESULTS_FILE.exists():
        print("\nBaseline Step 5 results not found; comparison columns will be skipped.")
        return None
    df = pd.read_excel(BASELINE_RESULTS_FILE)
    return {
        "tfidf": df.get("TF-IDF (CV)"),
        "bert": df.get("BERT (CV)"),
    }


def run_tfidf_augmented(df_real, df_synth):
    print("\n" + "=" * 60)
    print("  MODEL A: TF-IDF + LOGREG WITH SYNTHETIC URGENT AUGMENTATION")
    print("=" * 60)

    texts = df_real["text"].to_numpy(dtype=str)
    labels = df_real["gold"].to_numpy(dtype=str)
    synth_texts = df_synth["text"].to_list()
    synth_labels = df_synth["gold"].to_list()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = [None] * len(df_real)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        train_texts = texts[train_idx].tolist() + synth_texts
        train_labels = labels[train_idx].tolist() + synth_labels
        val_texts = texts[val_idx].tolist()

        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        clf.fit(X_train, train_labels)
        fold_preds = clf.predict(X_val)
        for i, global_idx in enumerate(val_idx):
            all_preds[global_idx] = fold_preds[i]
        print(f"  Fold {fold}/5 complete — train(real+synth)={len(train_texts)}, val(real)={len(val_idx)}")

    return evaluate(labels, all_preds, "TF-IDF + LogReg (augmented train only, 5-fold CV)"), all_preds


def run_bert_augmented(df_real, df_synth):
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        print("\nPyTorch/Transformers not installed. Skipping augmented BERT.")
        return None

    print("\n" + "=" * 60)
    print("  MODEL B: BERT WITH SYNTHETIC URGENT AUGMENTATION")
    print("=" * 60)

    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 8
    EPOCHS = 6
    LR = 1e-5
    GRAD_ACCUM = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts_real = df_real["text"].tolist()
    labels_real = [LABEL2ID[g] for g in df_real["gold"]]
    synth_texts = df_synth["text"].tolist()
    synth_labels = [LABEL2ID[g] for g in df_synth["gold"]]

    class EmailDataset(Dataset):
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

    def compute_weights(label_ids):
        from collections import Counter

        counts = Counter(label_ids)
        n = len(label_ids)
        raw_weights = [n / (3 * counts[i]) for i in range(3)]
        capped_weights = [min(w, 3.0) for w in raw_weights]
        return torch.tensor(capped_weights, dtype=torch.float).to(device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = [None] * len(texts_real)
    all_probs = [None] * len(texts_real)
    texts_np = np.array(texts_real)
    labels_np = np.array(labels_real)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_np, labels_np), 1):
        train_texts = [texts_real[i] for i in train_idx] + synth_texts
        train_labels = [labels_real[i] for i in train_idx] + synth_labels
        val_texts = [texts_real[i] for i in val_idx]
        val_labels = [labels_real[i] for i in val_idx]

        weights = compute_weights(train_labels)
        print(
            f"\n  Fold {fold}/5 — train(real+synth)={len(train_texts)}, val(real)={len(val_texts)}"
        )
        print(
            f"  Loss weights: URGENT={weights[0]:.2f} ACTION={weights[1]:.2f} INFO={weights[2]:.2f}"
        )

        train_ds = EmailDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_ds = EmailDataset(val_texts, val_labels, tokenizer, MAX_LEN)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(device)

        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_dl) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            optimizer.zero_grad()
            for step, batch in enumerate(train_dl):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, batch_labels) / GRAD_ACCUM
                loss.backward()
                total_loss += loss.item() * GRAD_ACCUM
                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            print(f"    Epoch {epoch+1}/{EPOCHS} — loss: {total_loss / len(train_dl):.4f}")

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

    y_true = [ID2LABEL[l] for l in labels_real]
    results = evaluate(y_true, all_preds, "BERT (augmented train only, 5-fold CV)")

    print("\n  Training final augmented BERT model for saving...")
    full_texts = texts_real + synth_texts
    full_labels = labels_real + synth_labels
    full_weights = compute_weights(full_labels)
    full_ds = EmailDataset(full_texts, full_labels, tokenizer, MAX_LEN)
    full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=full_weights)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=0.01)
    final_model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for step, batch in enumerate(full_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].to(device)
            outputs = final_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(full_dl):
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

    os.makedirs(AUGMENTED_BERT_OUTPUT_DIR, exist_ok=True)
    final_model.save_pretrained(AUGMENTED_BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(AUGMENTED_BERT_OUTPUT_DIR)
    print(f"  Augmented BERT model saved → {AUGMENTED_BERT_OUTPUT_DIR}")

    return results, np.array(all_probs), all_preds


def save_augmented_results(df_real, baseline_preds, tfidf_aug_preds, bert_aug_result=None):
    os.makedirs(AUGMENTED_RESULTS_FILE.parent, exist_ok=True)
    out = df_real[["subject", "body", "gold"]].copy()
    out.columns = ["Subject", "Body", "Gold Label"]

    if baseline_preds is not None:
        if baseline_preds["tfidf"] is not None:
            out["Baseline TF-IDF (CV)"] = baseline_preds["tfidf"]
        if baseline_preds["bert"] is not None:
            out["Baseline BERT (CV)"] = baseline_preds["bert"]

    out["Augmented TF-IDF (CV)"] = tfidf_aug_preds
    out["Augmented TF-IDF Correct"] = (
        out["Gold Label"].astype(str).str.strip().str.upper()
        == out["Augmented TF-IDF (CV)"].astype(str).str.strip().str.upper()
    ).map({True: "✓", False: "✗"})

    summary_rows = []
    if baseline_preds is not None and baseline_preds["tfidf"] is not None:
        baseline_tfidf = baseline_preds["tfidf"].astype(str).str.strip().str.upper().tolist()
        real_labels = out["Gold Label"].astype(str).str.strip().str.upper().tolist()
        baseline_tfidf_acc = sum(a == b for a, b in zip(real_labels, baseline_tfidf)) / len(real_labels)
        summary_rows.append({"Model": "Baseline TF-IDF", "Accuracy %": round(baseline_tfidf_acc * 100, 2)})

    aug_tfidf_acc = (
        out["Gold Label"].astype(str).str.strip().str.upper()
        == out["Augmented TF-IDF (CV)"].astype(str).str.strip().str.upper()
    ).mean()
    summary_rows.append({"Model": "Augmented TF-IDF", "Accuracy %": round(aug_tfidf_acc * 100, 2)})

    bert_probs = None
    if bert_aug_result is not None:
        bert_results, bert_probs, bert_preds = bert_aug_result
        out["Augmented BERT (CV)"] = bert_preds
        out["Augmented BERT Correct"] = (
            out["Gold Label"].astype(str).str.strip().str.upper()
            == out["Augmented BERT (CV)"].astype(str).str.strip().str.upper()
        ).map({True: "✓", False: "✗"})
        out["Augmented_BERT_P_URGENT"] = bert_probs[:, 0].round(3)
        out["Augmented_BERT_P_ACTION"] = bert_probs[:, 1].round(3)
        out["Augmented_BERT_P_INFO"] = bert_probs[:, 2].round(3)
        summary_rows.append({"Model": "Augmented BERT", "Accuracy %": round(bert_results["accuracy"] * 100, 2)})
        if baseline_preds is not None and baseline_preds["bert"] is not None:
            baseline_bert = baseline_preds["bert"].astype(str).str.strip().str.upper().tolist()
            real_labels = out["Gold Label"].astype(str).str.strip().str.upper().tolist()
            baseline_bert_acc = sum(a == b for a, b in zip(real_labels, baseline_bert)) / len(real_labels)
            summary_rows.append({"Model": "Baseline BERT", "Accuracy %": round(baseline_bert_acc * 100, 2)})

    with pd.ExcelWriter(AUGMENTED_RESULTS_FILE, engine="xlsxwriter") as writer:
        workbook = writer.book
        out.to_excel(writer, index=False, sheet_name="Results")
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")
        df_synth = load_synthetic_urgent()
        df_synth[["subject", "body", "gold"]].to_excel(writer, index=False, sheet_name="Synthetic_URGENT")

        worksheet = writer.sheets["Results"]
        green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

        for col_name in [c for c in out.columns if c.endswith("Correct")]:
            col_idx = out.columns.get_loc(col_name)
            for row_idx in range(1, len(out) + 1):
                val = out.iloc[row_idx - 1, col_idx]
                worksheet.write(row_idx, col_idx, val, green if val == "✓" else red)

    print(f"\nSaved augmentation comparison → {AUGMENTED_RESULTS_FILE}")


if __name__ == "__main__":
    gold_df = load_gold_data()
    synth_df = load_synthetic_urgent()
    baseline_preds = load_baseline_predictions()

    tfidf_aug_results, tfidf_aug_preds = run_tfidf_augmented(gold_df, synth_df)
    bert_aug_result = run_bert_augmented(gold_df, synth_df)

    save_augmented_results(gold_df, baseline_preds, tfidf_aug_preds, bert_aug_result)

    print("\n=== AUGMENTATION SUMMARY ===")
    print(f"  Augmented TF-IDF (5-fold CV): {tfidf_aug_results['accuracy'] * 100:.2f}%")
    if bert_aug_result is not None:
        print(f"  Augmented BERT (5-fold CV):   {bert_aug_result[0]['accuracy'] * 100:.2f}%")
    if baseline_preds is not None and baseline_preds["tfidf"] is not None:
        baseline_tfidf = baseline_preds["tfidf"].astype(str).str.strip().str.upper().tolist()
        gold = gold_df["gold"].astype(str).str.strip().str.upper().tolist()
        acc = sum(a == b for a, b in zip(gold, baseline_tfidf)) / len(gold)
        print(f"  Baseline TF-IDF (saved):      {acc * 100:.2f}%")
    if baseline_preds is not None and baseline_preds["bert"] is not None:
        baseline_bert = baseline_preds["bert"].astype(str).str.strip().str.upper().tolist()
        gold = gold_df["gold"].astype(str).str.strip().str.upper().tolist()
        acc = sum(a == b for a, b in zip(gold, baseline_bert)) / len(gold)
        print(f"  Baseline BERT (saved):        {acc * 100:.2f}%")
