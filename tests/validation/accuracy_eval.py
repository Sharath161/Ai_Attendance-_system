from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate(predictions_csv: Path, threshold: float) -> None:
    frame = pd.read_csv(predictions_csv)
    required = {"actual_student_id", "predicted_student_id", "similarity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["actual_known"] = frame["actual_student_id"].notna() & (frame["actual_student_id"] != "")
    frame["predicted_known"] = (
        frame["predicted_student_id"].notna()
        & (frame["predicted_student_id"] != "")
        & (frame["similarity"] >= threshold)
    )
    frame["correct_identity"] = frame["actual_student_id"] == frame["predicted_student_id"]
    frame["accepted_correct_match"] = frame["predicted_known"] & frame["actual_known"] & frame["correct_identity"]

    y_true = frame["actual_known"].astype(int)
    y_pred = frame["predicted_known"].astype(int)

    print(f"threshold={threshold}")
    print(f"precision={precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall={recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1={f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("confusion_matrix=[[tn, fp], [fn, tp]]")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["unknown", "known"], zero_division=0))

    false_accepts = frame[frame["predicted_known"] & ~frame["accepted_correct_match"]]
    if not false_accepts.empty:
        print("false_accept_count=" + str(len(false_accepts)))
        print(false_accepts[["actual_student_id", "predicted_student_id", "similarity"]].head(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate face attendance threshold and F1 score.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.65)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.predictions, args.threshold)
