# diagnose.py
"""
Shows misclassified images (FN/FP) and an explanation based on metric contributions.

Usage:
  python diagnose.py --csv metrics.csv --model model.json --top 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_model(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    for k in ("feature_cols", "mu", "sg", "w", "b"):
        if k not in obj:
            raise ValueError(f"model.json missing key '{k}'")
    obj["mu"] = np.array(obj["mu"], dtype=np.float64)
    obj["sg"] = np.array(obj["sg"], dtype=np.float64)
    obj["w"]  = np.array(obj["w"],  dtype=np.float64)
    obj["b"]  = float(obj["b"])
    obj["threshold"] = float(obj.get("threshold", 0.5))
    return obj


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def load_csv(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list: list[list[float]] = []
    y_list: list[int] = []
    files: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        need = ["file", "label"] + feature_cols
        missing = [c for c in need if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        for row in r:
            try:
                y = int(row["label"])
            except Exception:
                continue

            feats: list[float] = []
            ok = True
            for c in feature_cols:
                v = row.get(c, "")
                if v is None or v == "":
                    ok = False
                    break
                try:
                    fv = float(v)
                except Exception:
                    ok = False
                    break
                if not np.isfinite(fv):
                    ok = False
                    break
                feats.append(fv)

            if not ok:
                continue

            files.append(row["file"])
            y_list.append(y)
            X_list.append(feats)

    return np.asarray(X_list, dtype=np.float64), np.asarray(y_list, dtype=np.int64), files


def predict_proba_lr(X: np.ndarray, model: dict) -> np.ndarray:
    mu = model["mu"]
    sg = np.where(model["sg"] == 0, 1.0, model["sg"])
    w  = model["w"]
    b  = model["b"]
    Z = (X - mu) / sg
    logits = Z @ w + b
    return sigmoid(logits)


def top_contribs(x: np.ndarray, model: dict, k: int = 5) -> list[dict]:
    cols = model["feature_cols"]
    mu = model["mu"]
    sg = np.where(model["sg"] == 0, 1.0, model["sg"])
    w  = model["w"]

    z = (x - mu) / sg
    contrib = w * z
    idx = np.argsort(np.abs(contrib))[::-1][:k]

    out = []
    for i in idx:
        out.append({
            "metric": cols[i],
            "value": float(x[i]),
            "z": float(z[i]),
            "weight": float(w[i]),
            "contribution": float(contrib[i]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metrics.csv")
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--show-contribs", action="store_true", help="Print top-5 contributions for each error")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    cols = list(model["feature_cols"])
    thr = float(model["threshold"])

    X, y, files = load_csv(Path(args.csv), cols)
    probs = predict_proba_lr(X, model)
    preds = (probs >= thr).astype(np.int64)

    errors = np.where(preds != y)[0]
    fn_idx = [i for i in errors if y[i] == 1 and preds[i] == 0]  # AI -> Real
    fp_idx = [i for i in errors if y[i] == 0 and preds[i] == 1]  # Real -> AI

    # Sort by most confident errors
    fn_idx.sort(key=lambda i: probs[i])                # lowest P(AI)
    fp_idx.sort(key=lambda i: probs[i], reverse=True)  # highest P(AI)

    print(f"Total rows (valid)   : {len(y)}")
    print(f"Total errors         : {len(errors)}")
    print(f"FN (AI→Real)         : {len(fn_idx)}")
    print(f"FP (Real→AI)         : {len(fp_idx)}")
    print(f"Threshold            : {thr}\n")

    def print_block(title: str, idxs: list[int]):
        print("=" * 90)
        print(title)
        print("=" * 90)
        for i in idxs[:args.top]:
            print(f"{files[i]} | true={'AI' if y[i]==1 else 'Real'} pred={'AI' if preds[i]==1 else 'Real'} "
                  f"| p_ai={probs[i]:.4f}")
            if args.show_contribs:
                tc = top_contribs(X[i], model, k=5)
                for t in tc:
                    print(f"   {t['metric']:<24} contrib={t['contribution']:+.4f}  z={t['z']:+.3f}  x={t['value']:.4f}")
        print()

    print_block(f"TOP {args.top} FN: AI images classified as Real", fn_idx)
    print_block(f"TOP {args.top} FP: Real images classified as AI", fp_idx)


if __name__ == "__main__":
    main()