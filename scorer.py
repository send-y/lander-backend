# scorer.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metric_tool import compute_metrics


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-z))


def load_model(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("feature_cols", "mu", "sg", "w", "b"):
        if k not in obj:
            raise ValueError(f"Model file missing key: {k}")
    return obj


def score_image(image_path: str | Path, model_path: str | Path, z_clip: float = 6.0, sg_floor: float = 1e-3) -> dict:
    model = load_model(model_path)
    m = compute_metrics(image_path)

    cols = model["feature_cols"]
    x = np.array([float(getattr(m, c)) for c in cols], dtype=np.float64)

    mu = np.array(model["mu"], dtype=np.float64)
    sg = np.array(model["sg"], dtype=np.float64)
    w  = np.array(model["w"], dtype=np.float64)
    b  = float(model["b"])
    thr = float(model.get("threshold", 0.5))

    # robust normalization
    sg = np.maximum(sg, float(sg_floor))
    z = (x - mu) / sg
    z = np.clip(z, -float(z_clip), float(z_clip))

    logit = float(z @ w + b)
    p_ai = float(sigmoid(logit))

    # contributions to logit
    contrib = w * z
    idx = np.argsort(np.abs(contrib))[::-1][:5]
    top = [{
        "metric": cols[i],
        "value": float(x[i]),
        "z": float(z[i]),
        "weight": float(w[i]),
        "contribution": float(contrib[i]),
    } for i in idx]

    return {
        "image": str(image_path),
        "model": str(model_path),
        "p_ai": p_ai,
        "label": "ai" if p_ai >= thr else "real",
        "threshold": thr,
        "logit": logit,
        "top_contributions": top,
        "metrics": {k: float(getattr(m, k)) for k in cols},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score an image (AI vs Real).")
    ap.add_argument("image", help="Path to image")
    ap.add_argument("--model", default="model.json", help="Path to model.json")
    ap.add_argument("--verbose", action="store_true", help="Print metrics")
    ap.add_argument("--explain", action="store_true", help="Print top-5 metric contributions")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    ap.add_argument("--z-clip", type=float, default=6.0, help="Clip z-scores to [-z_clip, z_clip]")
    ap.add_argument("--sg-floor", type=float, default=1e-3, help="Minimum std dev for normalization")
    args = ap.parse_args()

    res = score_image(args.image, args.model, z_clip=args.z_clip, sg_floor=args.sg_floor)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    print(f"Image: {res['image']}")
    print(f"Model: {res['model']}")
    print(f"p_ai : {res['p_ai']:.4f} (thr={res['threshold']})")
    print(f"Pred : {res['label']}")

    if args.explain:
        print("\nTop contributions (to logit):")
        for t in res["top_contributions"]:
            print(f"  {t['metric']:<24} contrib={t['contribution']:+.4f}  z={t['z']:+.3f}  x={t['value']:.6f}  w={t['weight']:+.4f}")

    if args.verbose:
        print("\nMetrics:")
        for k, v in res["metrics"].items():
            print(f"  {k:<24} {v:.6f}")


if __name__ == "__main__":
    main()