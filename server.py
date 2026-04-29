# server.py
from __future__ import annotations

import json
import tempfile
import csv
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from metric_tool import compute_metrics


app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("model.json")
CSV_PATH = Path("analyses.csv")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-z))


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"model.json не знайдено: {path}")

    m = json.loads(path.read_text(encoding="utf-8"))

    for k in ("feature_cols", "type"):
        if k not in m:
            raise ValueError(f"model.json missing key '{k}'")

    m["threshold"] = float(m.get("threshold", 0.5))

    if m["type"] == "logreg_numpy":
        m["mu"] = np.array(m["mu"], dtype=np.float64)
        m["sg"] = np.array(m["sg"], dtype=np.float64)
        m["w"] = np.array(m["w"], dtype=np.float64)
        m["b"] = float(m["b"])
        m["z_clip"] = float(m.get("z_clip", 6.0))
        m["sg_floor"] = float(m.get("sg_floor", 1e-3))

    elif m["type"] == "random_forest":
        m["z_clip"] = None
        m["sg_floor"] = None

    return m


def predict_rf(x: np.ndarray, model: dict) -> tuple[float, float, list[dict]]:
    trees = model["trees"]
    cols = model["feature_cols"]
    probs = []

    for tree in trees:
        node = 0
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]

        while children_left[node] != -1:
            if x[feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]

        v = value[node][0]
        total = sum(v)
        p1 = v[1] / total if total > 0 else 0.5
        probs.append(p1)

    p_ai = float(np.mean(probs))
    p_ai_safe = float(np.clip(p_ai, 1e-9, 1 - 1e-9))
    logit = float(np.log(p_ai_safe / (1 - p_ai_safe)))

    votes = np.zeros(len(cols))

    for tree in trees:
        node = 0
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]

        while children_left[node] != -1:
            votes[feature[node]] += 1

            if x[feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]

    idx = np.argsort(votes)[::-1][:5]

    top = [
        {
            "metric": cols[i],
            "value": float(x[i]),
            "votes": float(votes[i]),
        }
        for i in idx
    ]

    return p_ai, logit, top


MODEL = load_model(MODEL_PATH)
COLS = list(MODEL["feature_cols"])


def predict_with_explain(x: np.ndarray) -> tuple[float, float, list[dict]]:
    if MODEL["type"] == "random_forest":
        return predict_rf(x, MODEL)

    mu = MODEL["mu"]
    sg = np.maximum(MODEL["sg"], MODEL["sg_floor"])
    w = MODEL["w"]
    b = MODEL["b"]

    z = (x - mu) / sg
    z = np.clip(z, -MODEL["z_clip"], MODEL["z_clip"])

    contrib = w * z
    logit = float(z @ w + b)
    p_ai = sigmoid(logit)

    idx = np.argsort(np.abs(contrib))[::-1][:5]

    top = [
        {
            "metric": COLS[i],
            "value": float(x[i]),
            "z": float(z[i]),
            "weight": float(w[i]),
            "contribution": float(contrib[i]),
        }
        for i in idx
    ]

    return p_ai, logit, top


@app.route("/api/health", methods=["GET"])
def health():
    resp = {
        "status": "ok",
        "model_type": MODEL.get("type", "unknown"),
        "modelVersion": MODEL.get("version", "unknown"),
        "n_features": len(COLS),
        "threshold": MODEL.get("threshold", 0.5),
    }

    if MODEL.get("type") == "logreg_numpy":
        resp["z_clip"] = MODEL.get("z_clip")
        resp["sg_floor"] = MODEL.get("sg_floor")

    if MODEL.get("type") == "random_forest":
        resp["n_estimators"] = MODEL.get("n_estimators")

    return jsonify(resp)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({
            "error": "Файл не знайдено в запиті (поле має називатися 'image')"
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Файл не вибрано"}), 400

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTS:
        return jsonify({
            "error": f"Непідтримуваний формат: {ext}. Дозволено: {sorted(ALLOWED_EXTS)}"
        }), 400

    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            file.save(tmp_path)

        m = compute_metrics(tmp_path)

        x = np.array(
            [float(getattr(m, c)) for c in COLS],
            dtype=np.float64
        )

        p_ai, logit, top = predict_with_explain(x)

        threshold = MODEL["threshold"]
        label = "AI-generated" if p_ai >= threshold else "Real photo"

        metrics_dict = {
            c: round(float(getattr(m, c)), 6)
            for c in COLS
        }

        extras = {}

        for k in (
            "metadata_flag",
            "exif_present",
            "orig_width",
            "orig_height",
            "file_size_kb",
            "file_bpp",
        ):
            if hasattr(m, k):
                v = getattr(m, k)

                if isinstance(v, (int, float, np.integer, np.floating)):
                    extras[k] = float(v)
                else:
                    extras[k] = v

        return jsonify({
            "probability": round(float(p_ai), 4),
            "label": label,
            "threshold": float(threshold),
            "logit": round(float(logit), 6),
            "top_contributions": top,
            "metrics": metrics_dict,
            "extras": extras,
            "modelVersion": MODEL.get("version", MODEL.get("type", "unknown")),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.route("/api/save-feedback", methods=["POST"])
def save_feedback():
    data = request.get_json(silent=True) or {}

    fieldnames = [
        "analysisId",
        "userId",
        "userEmail",
        "fileName",
        "fileType",
        "fileSize",
        "label",
        "originalLabel",
        "percent",
        "probability",
        "modelVersion",
        "isCorrect",
        "feedbackComment",
        "imageUrl",
        "feedbackSavedAt",
    ]

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            key: data.get(key, "")
            for key in fieldnames
        })

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)