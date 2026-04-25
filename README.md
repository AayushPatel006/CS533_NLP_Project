# CS533 NLP Project

This repository already contains most of the project pipeline. The main thing missing was a clean, portable way for all three teammates to run it from the repo instead of from one person's desktop paths.

## Project Goal

Classify Enron emails into:

- `URGENT`
- `ACTION`
- `INFORMATION`

The repo currently supports:

1. data cleaning / structuring
2. weak supervision with labeling functions
3. weighted LF evaluation
4. confusion-matrix error analysis
5. Snorkel label model
6. TF-IDF baseline and BERT evaluation on gold labels

## Recommended Final Pipeline

Run these files in order from the repository root:

```bash
pip install -r LF/improvements_aayush/requirements.txt
```

Step 1:

```bash
python LF/improvements_aayush/step1_step2_weighted_lfs_round_3.py
```

Output:
`LF/improvements_aayush/results/Weighted_LF/Batch_Weighted_LF_Results_round_3_portable.xlsx`

Step 2:

```bash
python LF/improvements_aayush/step3_confusion_matrix_analysis.py
```

Output:
`LF/improvements_aayush/results/Step3_Error_Analysis/Step3_Error_Analysis_portable.xlsx`

Step 3:

```bash
python LF/improvements_aayush/step4_snorkel_label_model.py
```

Output:
`LF/improvements_aayush/results/Step4_Snorkel_Results/Step4_Snorkel_Results_portable.xlsx`

Step 4:

```bash
python LF/improvements_aayush/step5_bert_classifier.py
```

Outputs:

- `LF/improvements_aayush/results/Step5_BERT/Step5_BERT_Results_portable.xlsx`
- `LF/improvements_aayush/results/bert_model/`

Optional augmentation experiment:

```bash
python LF/improvements_aayush/step5_urgent_augmentation_experiment.py
```

This trains TF-IDF and BERT with a small synthetic `URGENT` set added to the
training folds only, while still evaluating on the original gold-labelled
emails.

## Supporting Data Files

- Gold dataset: `dataset/Golden Dataset - 300 rows refined.xlsx`
- Example structured emails: `cleaning/enron_structured_first_60_rows.jsonl`

## Team Split Suggestion

- Teammate 1: Run Steps 1 to 3 and summarize LF behavior, confusion matrix, and weak-supervision results.
- Teammate 2: Run Step 4 and compare TF-IDF vs BERT using the gold labels.
- Teammate 3: Write the final report and slides using the generated Excel outputs, charts, and error examples.

## Notes

- The BERT script trains on gold labels, not Snorkel soft labels.
- `weak_labels/weak_labeling_func.py` now reads the sample JSONL already in the repo.
- Some older files in `LF/` still contain personal absolute paths because they appear to be earlier experiments; the portable pipeline above is the one to use for submission/demo work.
