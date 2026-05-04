from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LF.improvements_aayush import step1_step2_weighted_lfs_round_2 as round2_lf
from LF.improvements_aayush import step4_snorkel_label_model as snorkel_lf

INPUT_FILE = PROJECT_ROOT / "dataset" / "emails_10k_structured.csv"
OUTPUT_DIR = PROJECT_ROOT / "LF" / "improvements_aayush" / "results" / "Weak_Labels_10k"
ROUND2_OUTPUT = OUTPUT_DIR / "Round2_Weak_Labels_10k.csv"
SNORKEL_OUTPUT = OUTPUT_DIR / "Snorkel_Weak_Labels_10k.csv"


def run_round2_weighted_labels(df):
    predicted_labels = []
    score_rows = []
    for _, row in df.iterrows():
        label, scores, _ = round2_lf.predict_label(row)
        predicted_labels.append(label)
        score_rows.append(
            {
                "Round2_Predicted_Label": label,
                "Round2_Score_URGENT": scores.get(round2_lf.URGENT, 0.0),
                "Round2_Score_ACTION": scores.get(round2_lf.ACTION, 0.0),
                "Round2_Score_INFORMATION": scores.get(round2_lf.INFORMATION, 0.0),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(score_rows)], axis=1)


def run_snorkel_soft_labels(df):
    if not snorkel_lf.SNORKEL_AVAILABLE:
        raise RuntimeError("Snorkel is not installed. Install it before running Stage 3.")

    snorkel_functions = [
        snorkel_lf.LabelingFunction(name=fn.__name__, f=fn)
        for fn in snorkel_lf.ALL_LF_FUNCTIONS
    ]
    applier = snorkel_lf.PandasLFApplier(lfs=snorkel_functions)
    L_matrix = applier.apply(df)

    label_model = snorkel_lf.LabelModel(cardinality=3, verbose=True)
    label_model.fit(
        L_train=L_matrix,
        n_epochs=500,
        lr=0.001,
        log_freq=100,
        seed=42,
    )
    probs = label_model.predict_proba(L=L_matrix)
    preds = probs.argmax(axis=1)

    out = df.copy()
    out["Snorkel_Predicted_Label"] = [snorkel_lf.CLASS_NAMES[p] for p in preds]
    out["P_URGENT"] = probs[:, 0]
    out["P_ACTION"] = probs[:, 1]
    out["P_INFORMATION"] = probs[:, 2]
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    round2_out = run_round2_weighted_labels(df)
    round2_out.to_csv(ROUND2_OUTPUT, index=False)
    print(f"Saved Round 2 weak labels → {ROUND2_OUTPUT}")

    snorkel_out = run_snorkel_soft_labels(df)
    snorkel_out.to_csv(SNORKEL_OUTPUT, index=False)
    print(f"Saved Snorkel weak labels → {SNORKEL_OUTPUT}")


if __name__ == "__main__":
    main()
