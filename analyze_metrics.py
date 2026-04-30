# analyze_metrics.py
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metric_tool import FEATURE_COLS


# -------------------------
# CSV loading
# -------------------------

def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[list[float]] = []
    y_list: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)

        need = ["label"] + list(FEATURE_COLS)
        missing = [c for c in need if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        for row in r:
            try:
                y = int(row["label"])
            except Exception:
                continue

            feats: list[float] = []
            ok = True
            for c in FEATURE_COLS:
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

            X_list.append(feats)
            y_list.append(y)

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


# -------------------------
# Math / stats
# -------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    sp = np.sqrt((sa * sa + sb * sb) / 2.0) + 1e-12
    return float((mb - ma) / sp)


def print_statistics(X: np.ndarray, y: np.ndarray) -> None:
    real_X = X[y == 0]
    ai_X = X[y == 1]

    print("\n" + "=" * 82)
    print("METRIC STATISTICS")
    print("=" * 82)
    print(f"{'Metric':<22} {'Real mean':>10} {'Real std':>10} {'AI mean':>10} {'AI std':>10} {'Cohen d':>9}")
    print("─" * 82)

    for i, name in enumerate(FEATURE_COLS):
        r = real_X[:, i]
        a = ai_X[:, i]
        print(f"{name:<22} {r.mean():>10.4f} {r.std():>10.4f} {a.mean():>10.4f} {a.std():>10.4f} {cohen_d(r, a):>9.3f}")


# -------------------------
# Logistic Regression (numpy)
# -------------------------

@dataclass
class LRModel:
    mu: np.ndarray
    sg: np.ndarray
    w: np.ndarray
    b: float
    feature_cols: list[str]
    threshold: float = 0.5
    z_clip: float = 6.0
    sg_floor: float = 1e-3

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        sg = np.maximum(self.sg, self.sg_floor)
        Z = (X - self.mu) / sg
        Z = np.clip(Z, -self.z_clip, self.z_clip)
        return Z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self._normalize(X)
        return sigmoid(Z @ self.w + self.b)


def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    sg_floor: float,
    z_clip: float,
) -> LRModel:
    mu = X.mean(axis=0)
    sg = X.std(axis=0)
    sg = np.maximum(sg, sg_floor)

    Z = (X - mu) / sg
    Z = np.clip(Z, -z_clip, z_clip)

    n, d = Z.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    # Class weights (just in case)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    w0 = n / (2.0 * max(1, n0))
    w1 = n / (2.0 * max(1, n1))
    sw = np.where(y == 0, w0, w1).astype(np.float64)

    for _ in range(epochs):
        p = sigmoid(Z @ w + b)
        err = (p - y.astype(np.float64)) * sw

        dw = (Z.T @ err) / n + l2 * w
        db = float(err.mean())

        w -= lr * dw
        b -= lr * db

    return LRModel(
        mu=mu,
        sg=sg,
        w=w,
        b=b,
        feature_cols=list(FEATURE_COLS),
        threshold=0.5,
        z_clip=float(z_clip),
        sg_floor=float(sg_floor),
    )


def evaluate(model: LRModel, X: np.ndarray, y: np.ndarray) -> dict:
    p = model.predict_proba(X)
    pred = (p >= model.threshold).astype(np.int64)

    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)

    return {
        "accuracy": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(prec),
        "recall": float(rec),
    }


def save_model(model: LRModel, out_path: Path) -> None:
    obj = {
        "type": "logreg_numpy",
        "feature_cols": model.feature_cols,
        "mu": model.mu.tolist(),
        "sg": model.sg.tolist(),
        "w": model.w.tolist(),
        "b": float(model.b),
        "threshold": float(model.threshold),
        # Important: so the scorer/server knows how to normalize in the same way
        "z_clip": float(model.z_clip),
        "sg_floor": float(model.sg_floor),
    }
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--save", default="model.json")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--l2", type=float, default=1e-2)

    ap.add_argument("--sg-floor", type=float, default=1e-3, help="Minimum std for normalization (default 1e-3)")
    ap.add_argument("--z-clip", type=float, default=6.0, help="Clip z-scores to [-z_clip, z_clip] (default 6)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    print(f"Loading {csv_path} ...")
    X, y = load_csv(csv_path)

    n = len(y)
    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())
    print(f"Loaded: {n} images ({n_real} real, {n_ai} ai)")

    if n < 100:
        raise RuntimeError("Too little data after filtering. Check the CSV/columns.")

    print_statistics(X, y)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    split = int(n * 0.8)
    tr, te = idx[:split], idx[split:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    print("\nTraining Logistic Regression (numpy) with robust normalization...")
    print(f"  lr={args.lr} epochs={args.epochs} l2={args.l2} sg_floor={args.sg_floor} z_clip={args.z_clip}")

    model = train_lr(
        Xtr, ytr,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        sg_floor=args.sg_floor,
        z_clip=args.z_clip,
    )

    tr_m = evaluate(model, Xtr, ytr)
    te_m = evaluate(model, Xte, yte)

    print("\nTRAIN:", json.dumps(tr_m, indent=2, ensure_ascii=False))
    print("TEST :", json.dumps(te_m, indent=2, ensure_ascii=False))

    out_path = Path(args.save)
    save_model(model, out_path)
    print(f"\nModel saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()