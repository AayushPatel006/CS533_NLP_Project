"""
STEP 3: CONFUSION MATRIX + PER-CLASS ERROR ANALYSIS
=====================================================
Run this AFTER step1_step2_weighted_lfs.py to understand exactly WHERE
your errors are clustering. This tells you which new LFs to write next.

Outputs:
  1. Terminal: confusion matrix + top 10 misclassified emails per class
  2. Excel:    full error audit spreadsheet with mismatch type column
"""

import pandas as pd
import re
import os

# ─────────────────────────────────────────────────────────────────────
# IMPORT the predict function from your Step 1/2 file.
# If running standalone, paste the predict_label function here instead.
# ─────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

# We re-implement a minimal import shim so this file is self-contained.
# Just point it at the same CSV and it re-runs predictions internally.
# ─────────────────────────────────────────────────────────────────────

INPUT_FILE  = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/LF/improvements_aayush/results/Step3_Error_Analysis_round_1.xlsx"
OUTPUT_FILE = "/Users/aayushpatel/Desktop/Rutgers/Academics/Spring 2026/NLP/NLP Project/LF/improvements_aayush/results/Step3_Error_Analysis.xlsx"


ABSTAIN      = None
URGENT       = "URGENT"
ACTION       = "ACTION"
INFORMATION  = "INFORMATION"

# ── Paste your helpers here (or import from step1_step2) ──────────────

def normalize(text):
    if not text or not isinstance(text, str): return ""
    return re.sub(r"\s+", " ", text.lower()).strip()

def clean_body(body):
    body = re.sub(r'-----original message-----.*', '', body, flags=re.DOTALL)
    body = re.sub(r'---------------------- forwarded by.*', '', body, flags=re.DOTALL)
    body = re.sub(r'on .{5,50} wrote:.*', '', body, flags=re.DOTALL)
    return body

def contains_any(text, keywords):
    return any(k in text for k in keywords)

def regex_match(text, patterns):
    return any(re.search(p, text) for p in patterns)


# ─────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_confusion_matrix(actual_series, predicted_series, classes):
    """Returns a dict-of-dicts: matrix[actual][predicted] = count."""
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for actual, predicted in zip(actual_series, predicted_series):
        if actual in matrix and predicted in matrix:
            matrix[actual][predicted] += 1
    return matrix

def print_confusion_matrix(matrix, classes):
    col_width = 14
    print("\n" + "="*60)
    print("  CONFUSION MATRIX  (rows = Actual, cols = Predicted)")
    print("="*60)
    header = f"{'':14s}" + "".join(f"{c:>{col_width}}" for c in classes)
    print(header)
    print("-"*60)
    for actual in classes:
        row_str = f"{actual:<14s}"
        for predicted in classes:
            count = matrix[actual][predicted]
            marker = " ✓" if actual == predicted else "  "
            row_str += f"{str(count)+marker:>{col_width}}"
        print(row_str)
    print("="*60)

def print_per_class_stats(matrix, classes):
    print("\n  PER-CLASS BREAKDOWN")
    print("  " + "-"*40)
    for cls in classes:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in classes if other != cls)
        fn = sum(matrix[cls][other] for other in classes if other != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        print(f"  {cls:<12} | Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")
    print("  " + "-"*40)


# ─────────────────────────────────────────────────────────────────────
# ERROR CATEGORISATION
# ─────────────────────────────────────────────────────────────────────

def categorize_mismatch(actual, predicted):
    """Returns a human-readable mismatch label."""
    if actual == predicted:
        return "CORRECT"
    return f"{actual} → predicted as {predicted}"

def get_top_errors(df, actual_col, predicted_col, actual_class, predicted_as, n=10):
    """Return top-n rows where actual=actual_class but predicted=predicted_as."""
    mask = (
        (df[actual_col].str.strip().str.upper() == actual_class) &
        (df[predicted_col].str.strip().str.upper() == predicted_as)
    )
    subset = df[mask].copy()
    # Truncate body for readability in terminal
    subset['body_preview'] = subset['body'].apply(
        lambda x: str(x)[:200].replace('\n', ' ') + "..."
    )
    return subset[['subject', 'body_preview']].head(n)


# ─────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_error_analysis():
    # ── Load the OUTPUT from your step1/step2 script ──
    # If that file doesn't exist yet, fall back to re-running predictions inline.
    weighted_output = OUTPUT_FILE.replace("Step3_Error_Analysis", "Batch_Weighted_LF_Results").replace(".xlsx", ".xlsx")

    if os.path.exists(weighted_output):
        print(f"Loading weighted LF results from: {weighted_output}")
        df = pd.read_excel(weighted_output)
    elif os.path.exists(INPUT_FILE):
        print("Weighted output not found — loading raw CSV and using simple predictions.")
        print("Run step1_step2_weighted_lfs.py first for best results.\n")
        dfs = pd.read_excel(INPUT_FILE, sheet_name=None)
        df = pd.concat(dfs.values(), ignore_index=True)
        # Fallback: use a trivial baseline so analysis still runs
        df['Predicted Label'] = "INFORMATION"
    else:
        print(f"Error: Neither input nor weighted output found.")
        return

    # ── Filter TIEs ──
    df = df[df['Final Label'].str.strip().str.upper() != "TIE"].copy()
    df['_actual']    = df['Final Label'].str.strip().str.upper()
    df['_predicted'] = df['Predicted Label'].str.strip().str.upper()
    df['Mismatch Type'] = df.apply(
        lambda r: categorize_mismatch(r['_actual'], r['_predicted']), axis=1
    )

    classes = [URGENT, ACTION, INFORMATION]
    matrix  = build_confusion_matrix(df['_actual'], df['_predicted'], classes)

    # ── Terminal output ──
    correct = (df['_actual'] == df['_predicted']).sum()
    total   = len(df)
    print("\n" + "="*60)
    print(f"  OVERALL ACCURACY: {correct}/{total} = {(correct/total)*100:.2f}%")
    print_confusion_matrix(matrix, classes)
    print_per_class_stats(matrix, classes)

    # ── Top error pairs ──
    error_pairs = [
        (URGENT,      ACTION),
        (URGENT,      INFORMATION),
        (ACTION,      URGENT),
        (ACTION,      INFORMATION),
        (INFORMATION, URGENT),
        (INFORMATION, ACTION),
    ]

    print("\n" + "="*60)
    print("  TOP MISCLASSIFICATIONS (for writing new LFs)")
    print("="*60)
    for actual_cls, predicted_cls in error_pairs:
        count = matrix[actual_cls][predicted_cls]
        if count == 0:
            continue
        print(f"\n  ── {actual_cls} emails predicted as {predicted_cls} ({count} cases) ──")
        top_errors = get_top_errors(df, '_actual', '_predicted', actual_cls, predicted_cls, n=5)
        for i, (_, err_row) in enumerate(top_errors.iterrows(), 1):
            print(f"  [{i}] Subject: {err_row['subject']}")
            print(f"      Body: {err_row['body_preview']}")
            print()

    # ── Save full error audit to Excel ──
    error_df = df[df['_actual'] != df['_predicted']].copy()
    error_df = error_df[['subject', 'body', 'Final Label', 'Predicted Label', 'Mismatch Type']]

    writer    = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    workbook  = writer.book

    # Sheet 1: All errors
    error_df.to_excel(writer, index=False, sheet_name='All_Errors')

    # Sheet 2: Confusion matrix as a table
    matrix_rows = []
    for actual_cls in classes:
        row = {'Actual \\ Predicted': actual_cls}
        for pred_cls in classes:
            row[pred_cls] = matrix[actual_cls][pred_cls]
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_excel(writer, index=False, sheet_name='Confusion_Matrix')

    # Sheet 3: One tab per error pair
    for actual_cls, predicted_cls in error_pairs:
        subset = error_df[
            (error_df['Final Label'].str.strip().str.upper() == actual_cls) &
            (error_df['Predicted Label'].str.strip().str.upper() == predicted_cls)
        ]
        if len(subset) > 0:
            sheet_name = f"{actual_cls[:3]}_as_{predicted_cls[:3]}"
            subset.to_excel(writer, index=False, sheet_name=sheet_name)

    # Colour the Mismatch Type column on the All_Errors sheet
    worksheet = writer.sheets['All_Errors']
    red_fmt   = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    mismatch_col = error_df.columns.get_loc('Mismatch Type')
    for row_idx in range(1, len(error_df) + 1):
        val = str(error_df.iloc[row_idx-1]['Mismatch Type'])
        worksheet.write(row_idx, mismatch_col, val, red_fmt)

    writer.close()
    print(f"\n  Full error audit saved → {OUTPUT_FILE}")
    print("  Tabs: All_Errors | Confusion_Matrix | URG_as_ACT | etc.\n")


if __name__ == "__main__":
    run_error_analysis()
