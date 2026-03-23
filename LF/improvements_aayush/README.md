# NLP Email Classification — Steps 1–4
## How to run these files

### Installation
```bash
pip install pandas openpyxl xlsxwriter snorkel
```

---

### Execution order

```
Step 1 & 2  →  Step 3  →  Step 4
```

---

### Step 1 & 2 — `step1_step2_weighted_lfs.py`

What it does:
- Replaces your binary 1/ABSTAIN voting with a **weighted confidence score**
- Fixes the three biggest misclassification patterns:
  1. INFO gatekeeper is now soft (adds INFO weight) not a hard block
  2. Scheduling alone is ACTION; scheduling + time pressure is URGENT
  3. Negation filter prevents "I will send" from being classified as ACTION
- Subject lines get 1.5× weight multiplier
- Outputs a colour-coded Excel with URGENT/ACTION/INFO score columns

Update the `INPUT_FILE` and `OUTPUT_FILE` paths at the top before running.

```bash
python step1_step2_weighted_lfs.py
```

Expected output (rough estimate):
```
WEIGHTED LF PERFORMANCE REPORT
Total Evaluated  : ~97
Accuracy         : ~62–70%
```

---

### Step 3 — `step3_confusion_matrix_analysis.py`

What it does:
- Loads the Step 1/2 output Excel
- Prints a full confusion matrix to terminal
- Shows per-class Precision / Recall / F1
- Prints top 5 misclassified emails for EVERY error pair (URGENT→ACTION, ACTION→INFO, etc.)
- Saves an Excel with tabs: All_Errors | Confusion_Matrix | URG_as_ACT | etc.

```bash
python step3_confusion_matrix_analysis.py
```

**How to use the output:** Look at the misclassified emails tab by tab.
For each error bucket, ask: "What pattern do these emails share that 
a new LF could catch?" Then add that LF to Step 1/2 and re-run.

---

### Step 4 — `step4_snorkel_label_model.py`

What it does:
- Takes all 20 LFs and builds an LF matrix (N emails × 20 LFs)
- Uses Snorkel's `LabelModel` to learn LF accuracies automatically
  (no gold labels needed — it learns from the LF agreement patterns)
- Outputs probabilistic labels (P_URGENT, P_ACTION, P_INFORMATION per email)
- Falls back to majority vote if Snorkel isn't installed
- Shows an LF Coverage sheet so you can see which LFs fire most

```bash
pip install snorkel   # one-time
python step4_snorkel_label_model.py
```

Expected improvement over Step 1/2: +3–8% accuracy, because LabelModel
down-weights LFs that are frequently wrong and up-weights reliable ones.

---

### What to do after these four steps

Once Step 4 accuracy reaches ~65–70%:
1. Use the weak labels from Step 4 (`Predicted Label` column) as training data
2. Train a TF-IDF + Logistic Regression baseline (fast, gives you a floor)
3. Fine-tune a BERT/RoBERTa model on the weak labels
4. Evaluate all models on your 300-email gold set
5. Add sender metadata (role, thread position) for the personalised model

---

### File summary

| File | Purpose |
|------|---------|
| `step1_step2_weighted_lfs.py` | Weighted LF scoring + fixed gatekeeper |
| `step3_confusion_matrix_analysis.py` | Error analysis + confusion matrix |
| `step4_snorkel_label_model.py` | Snorkel LabelModel integration |
| `README.md` | This file |
