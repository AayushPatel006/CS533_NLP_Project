# CS533 NLP Project

This repository contains the code, data pipeline, and model experiments used for our email priority classification project.


## Project Goal

Classify Enron emails into:

- `URGENT`
- `ACTION`
- `INFORMATION`

The repository includes the following components:

1. data preparation and email structuring
2. weighted labeling-function pipelines across multiple rounds
3. confusion-matrix and error analysis
4. Snorkel-based weak supervision
5. TF-IDF + Logistic Regression baseline
6. gold-only BERT evaluation
7. large-scale weak-label generation on 10,000 Enron emails
8. weak-pretraining plus gold fine-tuning for the final BERT model
9. saved result files and trained model outputs

## How to Run the Project

Please run all commands from the repository root.
We recommend creating a virtual environment first.

### Step 1: Create and activate a virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r LF/improvements_aayush/requirements.txt
```

This installs the packages needed for the labeling-function pipeline, Snorkel,
TF-IDF baseline, and transformer experiments.

### Step 3: Run the weighted labeling-function pipeline

```bash
python LF/improvements_aayush/step1_step2_weighted_lfs_round_3.py
```

This script applies the final weighted labeling-function system and saves the
predicted labels and class scores.

Output:

- `LF/improvements_aayush/results/Weighted_LF/Batch_Weighted_LF_Results_round_3_portable.xlsx`

### Step 4: Run the confusion-matrix error analysis

```bash
python LF/improvements_aayush/step3_confusion_matrix_analysis.py
```

This script reads the weighted LF output from Step 2, computes the confusion
matrix, and saves the detailed error analysis.

Output:

- `LF/improvements_aayush/results/Step3_Error_Analysis/Step3_Error_Analysis_portable.xlsx`

### Step 5: Run the Snorkel label model

```bash
python LF/improvements_aayush/step4_snorkel_label_model.py
```

This script applies Snorkel to the labeling-function outputs and writes weak
labels and class probabilities.

Output:

- `LF/improvements_aayush/results/Step4_Snorkel_Results/Step4_Snorkel_Results_portable.xlsx`

### Step 6: Run the gold-label supervised baselines

```bash
python LF/improvements_aayush/step5_bert_classifier.py
```

This script evaluates:

- TF-IDF + Logistic Regression
- gold-only BERT

Outputs:

- `LF/improvements_aayush/results/Step5_BERT/Step5_BERT_Results_portable.xlsx`
- `LF/improvements_aayush/results/bert_model/`

### Step 7: Run the weak-supervision scaling pipeline

Run the following three scripts in order:

```bash
python LF/improvements_aayush/stage3_prepare_10k_structured_dataset.py
python LF/improvements_aayush/stage3_generate_weak_labels_10k.py
python LF/improvements_aayush/stage3_weak_pretrain_then_gold_finetune.py
```

These scripts:

1. prepare the structured 10,000-email Enron sample,
2. generate weak labels and Snorkel probabilities on that sample,
3. pretrain BERT on the weak labels and then fine-tune it on the 294-email gold set.

Outputs:

- `LF/improvements_aayush/results/Weak_Labels_10k/Round2_Weak_Labels_10k.csv`
- `LF/improvements_aayush/results/Weak_Labels_10k/Snorkel_Weak_Labels_10k.csv`
- `LF/improvements_aayush/results/Stage3_Weak_Pretrain/Stage3_Weak_Pretrain_Results.xlsx`
- `LF/improvements_aayush/results/bert_model_weak_pretrained/`

### Optional: Synthetic urgent augmentation experiment

```bash
python LF/improvements_aayush/step5_urgent_augmentation_experiment.py
```

This script adds a small synthetic `URGENT` set to the training folds only and
evaluates TF-IDF and BERT on the original gold-labelled data.

Output:

- `LF/improvements_aayush/results/Step5_BERT/Step5_Urgent_Augmentation_Results.xlsx`

## Required Data Files

The following files are needed for the main pipeline:

- `dataset/Golden Dataset - 300 rows refined.xlsx`
- `dataset/emails_10k_sample.csv`
- `dataset/emails_10k_structured.csv`

## Main Output Files

The most important output files are:

- `LF/improvements_aayush/results/Weighted_LF/Batch_Weighted_LF_Results_round_3_portable.xlsx`
- `LF/improvements_aayush/results/Step3_Error_Analysis/Step3_Error_Analysis_portable.xlsx`
- `LF/improvements_aayush/results/Step4_Snorkel_Results/Step4_Snorkel_Results_portable.xlsx`
- `LF/improvements_aayush/results/Step5_BERT/Step5_BERT_Results_portable.xlsx`
- `LF/improvements_aayush/results/Stage3_Weak_Pretrain/Stage3_Weak_Pretrain_Results.xlsx`

## Saved Models

The repository also includes saved model folders:

- `LF/improvements_aayush/results/bert_model/`
- `LF/improvements_aayush/results/bert_model_weak_pretrained/`
