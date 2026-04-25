"""
STEP 5: CLASSIFIER TRAINED ON GOLD LABELS
==========================================
KEY FIX FROM STEP 5 v1:
  Snorkel's soft labels are UNINFORMATIVE for URGENT:
    Mean P_URGENT for actual URGENT     = 0.238
    Mean P_URGENT for actual ACTION     = 0.242
    Mean P_URGENT for actual INFO       = 0.260
  All three classes get nearly identical P_URGENT scores.
  BERT trained on these learns to never predict URGENT.

THE SOLUTION:
  Train directly on the 294 GOLD labels instead of Snorkel soft labels.
  Use class_weight='balanced' to handle URGENT imbalance (18 vs 123 vs 153).
  Use 5-fold cross-validation to get honest accuracy estimates.

  294 gold labels is sufficient for BERT fine-tuning — this is standard
  practice for domain-specific classification with small labelled sets.

TWO MODELS:yes 
  1. TF-IDF + Logistic Regression — 5-fold CV on gold labels (fast baseline)
  2. BERT fine-tuned — 5-fold CV on gold labels with weighted loss
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# CONFIG — update these paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNORKEL_OUTPUT = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step4_Snorkel_Results"
    / "Step4_Snorkel_Results_portable.xlsx"
)
RESULTS_FILE = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "Step5_BERT"
    / "Step5_BERT_Results_portable.xlsx"
)
BERT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "LF"
    / "improvements_aayush"
    / "results"
    / "bert_model"
)

LABEL2ID = {"URGENT": 0, "ACTION": 1, "INFORMATION": 2}
ID2LABEL = {0: "URGENT", 1: "ACTION", 2: "INFORMATION"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_body(body):
    body = re.sub(r'-----original message-----.*', '', str(body), flags=re.DOTALL)
    body = re.sub(r'---------------------- forwarded by.*', '', body, flags=re.DOTALL)
    body = re.sub(r'on .{5,50} wrote:.*', '', body, flags=re.DOTALL)
    return body.strip()

def make_text(row):
    subj = str(row.get('subject', '') or row.get('Subject', '') or '')
    body = clean_body(row.get('body', '') or row.get('Body', '') or '')
    return f"{subj} [SEP] {body}"

def load_data():
    """Load Snorkel output. We use the email text + gold labels only."""
    print("Loading data from Snorkel output...")
    df = pd.read_excel(SNORKEL_OUTPUT, sheet_name='Snorkel_Results')
    df['text'] = df.apply(make_text, axis=1)
    df['gold'] = df['Final Label'].str.strip().str.upper()
    df = df[df['gold'] != 'TIE'].copy()
    df = df[df['gold'].isin(['URGENT', 'ACTION', 'INFORMATION'])].copy()
    print(f"  Loaded {len(df)} emails")
    print(f"  Gold label distribution:\n{df['gold'].value_counts().to_string()}")
    
    # Report why Snorkel soft labels are not used
    if 'P_URGENT' in df.columns:
        mean_p_by_class = {}
        print("\n  Snorkel P_URGENT by actual class (confirming soft labels are uninformative):")
        for cls in ['URGENT','ACTION','INFORMATION']:
            mean_p = df[df['gold']==cls]['P_URGENT'].mean()
            mean_p_by_class[cls] = mean_p
            print(f"    Actual {cls:<12}: mean P_URGENT = {mean_p:.3f}")
        mean_values = list(mean_p_by_class.values())
        print(f"  → All three classes have similarly low P_URGENT "
              f"({min(mean_values):.3f}–{max(mean_values):.3f})")
        print("  → Using GOLD labels for training instead.\n")
    return df

def evaluate(y_true, y_pred, model_name):
    classes = ['URGENT', 'ACTION', 'INFORMATION']
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total   = len(y_true)
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    for cls in classes:
        corr  = sum((np.array(y_true)==cls) & (np.array(y_pred)==cls))
        tot_c = sum(np.array(y_true)==cls)
        print(f"  {cls:<14}: {corr}/{tot_c} ({corr/tot_c*100 if tot_c else 0:.1f}%)")
    print(f"\n{classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=0)}")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':14} {'URGENT':>8} {'ACTION':>8} {'INFO':>8}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<14} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    return {'accuracy': correct/total, 'predictions': list(y_pred)}


# ─────────────────────────────────────────────
# MODEL 1: TF-IDF + LOGISTIC REGRESSION
# ─────────────────────────────────────────────

def run_tfidf_baseline(df):
    """
    Trained on GOLD labels with class_weight='balanced'.
    5-fold stratified cross-validation for honest evaluation.
    URGENT class weight ~5.4x corrects for the 18-email imbalance.
    """
    print("\n" + "="*55)
    print("  MODEL 1: TF-IDF + LOGISTIC REGRESSION")
    print("  (gold labels + balanced weights, 5-fold CV)")
    print("="*55)

    X      = df["text"].to_numpy(dtype=str)
    y_gold = df["gold"].to_numpy(dtype=str)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    X_tfidf = vectorizer.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs',
    )

    # Cross-val predictions — each email predicted on a fold it wasn't trained on
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_tfidf, y_gold, cv=cv)
    results = evaluate(y_gold, y_pred_cv, "TF-IDF + LogReg (gold labels, 5-fold CV)")

    # Refit on full data for the Excel output
    clf.fit(X_tfidf, y_gold)
    return vectorizer, clf, results, list(y_pred_cv)


# ─────────────────────────────────────────────
# MODEL 2: BERT FINE-TUNING ON GOLD LABELS
# ─────────────────────────────────────────────

def run_bert_classifier(df):
    """
    Fine-tune BERT on GOLD labels using weighted CrossEntropy loss.
    5-fold stratified cross-validation.
    URGENT receives ~5.4x higher loss weight to force the model to learn it.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        print("\n⚠️  PyTorch/Transformers not installed.")
        print("    Run: pip install torch transformers")
        print("    Skipping BERT.\n")
        return None

    print("\n" + "="*55)
    print("  MODEL 2: BERT FINE-TUNING")
    print("  (gold labels + weighted loss, 5-fold CV)")
    print("="*55)

    MODEL_NAME = "bert-base-uncased"
    MAX_LEN    = 256
    BATCH_SIZE = 8
    EPOCHS     = 6
    LR         = 1e-5   # lower LR improves stability with small dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts  = df['text'].tolist()
    labels = [LABEL2ID[g] for g in df['gold']]

    # Class weights: total / (n_classes * count_per_class)
    from collections import Counter
    counts = Counter(labels)
    n      = len(labels)
    # Cap weights at 3.0 — higher values cause BERT to collapse toward the rare class
    # (URGENT weight of 5.44 caused 280/294 emails to be predicted URGENT in v1)
    raw_weights = [n / (3 * counts[i]) for i in range(3)]
    capped_weights = [min(w, 3.0) for w in raw_weights]
    weights = torch.tensor(capped_weights, dtype=torch.float).to(device)
    print(f"  Loss weights: URGENT={weights[0]:.2f}  ACTION={weights[1]:.2f}  INFO={weights[2]:.2f}")

    class EmailDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts     = texts
            self.labels    = labels
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self): return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return {
                'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label':          torch.tensor(self.labels[idx], dtype=torch.long),
            }

    # ── 5-fold cross-validation ──
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds  = [None] * len(texts)
    all_probs  = [None] * len(texts)

    texts_np = np.array(texts)
    labels_np = np.array(labels)
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_np, labels_np)):
        print(f"\n  Fold {fold+1}/5 — train={len(train_idx)}, val={len(val_idx)}")

        train_texts  = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts    = [texts[i] for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        train_ds = EmailDataset(train_texts, train_labels, tokenizer, MAX_LEN)
        val_ds   = EmailDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3,
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        ).to(device)

        # Weighted cross-entropy — URGENT gets much higher loss weight
        loss_fn   = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_dl) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        GRAD_ACCUM = 2   # accumulate 2 steps → effective batch size = 16
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_dl):
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels   = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, batch_labels) / GRAD_ACCUM
                loss.backward()
                total_loss += loss.item() * GRAD_ACCUM
                if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            print(f"    Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(train_dl):.4f}")

        # Evaluate this fold
        model.eval()
        with torch.no_grad():
            val_offset = 0
            for batch in val_dl:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=1)
                batch_val_idx = val_idx[val_offset:val_offset + len(preds)]
                for i, global_idx in enumerate(batch_val_idx):
                    all_preds[global_idx] = ID2LABEL[preds[i]]
                    all_probs[global_idx] = probs[i]
                val_offset += len(preds)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Collect results
    y_true = [ID2LABEL[l] for l in labels]
    results = evaluate(y_true, all_preds, "BERT (gold labels, weighted loss, 5-fold CV)")

    # Save final model trained on ALL data
    print("\n  Training final model on all data for saving...")
    full_ds = EmailDataset(texts, labels, tokenizer, MAX_LEN)
    full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=0.01)
    final_model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for step, batch in enumerate(full_dl):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels   = batch['label'].to(device)
            outputs = final_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(full_dl):
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

    os.makedirs(BERT_OUTPUT_DIR, exist_ok=True)
    final_model.save_pretrained(BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(BERT_OUTPUT_DIR)
    print(f"  BERT model saved → {BERT_OUTPUT_DIR}")

    return results, np.array(all_probs), all_preds


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

def save_results(df, tfidf_cv_preds, bert_preds=None, bert_probs=None):
    os.makedirs(RESULTS_FILE.parent, exist_ok=True)
    out = df[['subject', 'body', 'gold']].copy() if 'subject' in df.columns else df[['Subject','Body','gold']].copy()
    out.columns = ['Subject', 'Body', 'Gold Label']
    out['TF-IDF (CV)'] = tfidf_cv_preds
    out['TF-IDF Correct'] = (out['Gold Label'] == out['TF-IDF (CV)']).map({True:'✓', False:'✗'})
    if bert_preds:
        out['BERT (CV)'] = bert_preds
        out['BERT Correct'] = (out['Gold Label'] == out['BERT (CV)']).map({True:'✓', False:'✗'})
    if bert_probs is not None:
        out['BERT_P_URGENT'] = bert_probs[:, 0].round(3)
        out['BERT_P_ACTION'] = bert_probs[:, 1].round(3)
        out['BERT_P_INFO']   = bert_probs[:, 2].round(3)

    writer    = pd.ExcelWriter(RESULTS_FILE, engine='xlsxwriter')
    workbook  = writer.book
    out.to_excel(writer, index=False, sheet_name='Results')
    worksheet   = writer.sheets['Results']
    fmt_correct = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    fmt_wrong   = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    tfidf_col = out.columns.get_loc('TF-IDF (CV)')
    for row_idx in range(1, len(out) + 1):
        gold = str(out.iloc[row_idx-1]['Gold Label']).strip().upper()
        pred = str(out.iloc[row_idx-1]['TF-IDF (CV)']).strip().upper()
        worksheet.write(row_idx, tfidf_col, out.iloc[row_idx-1]['TF-IDF (CV)'],
                        fmt_correct if gold == pred else fmt_wrong)
    writer.close()
    print(f"\n  Results saved → {RESULTS_FILE}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    # Model 1: TF-IDF baseline
    vectorizer, clf, tfidf_results, tfidf_cv_preds = run_tfidf_baseline(df)

    # Model 2: BERT
    bert_result = run_bert_classifier(df)
    if bert_result:
        bert_results, bert_probs, bert_preds = bert_result
        save_results(df, tfidf_cv_preds, bert_preds, bert_probs)
    else:
        save_results(df, tfidf_cv_preds)

    print("\n=== SUMMARY ===")
    print(f"  TF-IDF + LogReg (5-fold CV): {tfidf_results['accuracy']*100:.2f}%")
    if bert_result:
        print(f"  BERT (5-fold CV):            {bert_results['accuracy']*100:.2f}%")
    print("\n  Expected ranges:")
    print("  TF-IDF: 62–68%  |  BERT: 70–78%")
    print("\n  Next: personalised BERT with sender/thread metadata (Step 6)")
