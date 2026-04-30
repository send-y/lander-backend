# collect_metrics.py
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from metric_tool import compute_metrics, FEATURE_COLS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

EXTRA_COLS = [
    "metadata_flag",
    "exif_present",
    "orig_width",
    "orig_height",
    "orig_aspect",
    "file_size_kb",
    "file_bpp",
]


def iter_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def one_job(path: Path, label: int) -> dict:
    m = compute_metrics(path)

    row: dict = {"file": str(path), "label": int(label)}

    # FEATURES
    for c in FEATURE_COLS:
        v = getattr(m, c, None)
        if v is None:
            raise ValueError(f"metric '{c}' is None")
        row[c] = float(v)

    # EXTRAS
    for c in EXTRA_COLS:
        v = getattr(m, c, "")
        if isinstance(v, (int, float, np.integer, np.floating)):  # type: ignore[name-defined]
            row[c] = float(v)
        else:
            row[c] = v

    return row


def collect(real_dir: Path, ai_dir: Path, out_csv: Path, workers: int) -> None:
    real_files = iter_images(real_dir)
    ai_files = iter_images(ai_dir)

    jobs: list[tuple[Path, int]] = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]
    total = len(jobs)

    metric_cols = list(FEATURE_COLS) + EXTRA_COLS
    fieldnames = ["file", "label"] + metric_cols

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Real: {len(real_files)} from {real_dir}")
    print(f"AI  : {len(ai_files)} from {ai_dir}")
    print(f"Out : {out_csv}")
    print(f"Cols: {len(metric_cols)} metrics (+ file,label)")
    print(f"Workers: {workers}\n")

    done = 0
    err = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2job = {ex.submit(one_job, p, y): (p, y) for (p, y) in jobs}

            for fut in as_completed(fut2job):
                p, y = fut2job[fut]
                try:
                    row = fut.result()

                    for k in metric_cols:
                        if isinstance(row.get(k), float):
                            row[k] = round(row[k], 6)

                    w.writerow(row)

                except Exception as e:
                    err += 1
                    print(f"[ERROR] {p} (label={y}) -> {e}")

                done += 1
                if done % 200 == 0 or done == total:
                    print(f"Progress: {done}/{total}  errors={err}")

    print(f"\nDone: {out_csv}")
    print(f"Errors: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect metrics into CSV.")
    ap.add_argument("--real", required=True)
    ap.add_argument("--ai", required=True)
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = ap.parse_args()

    collect(Path(args.real), Path(args.ai), Path(args.out), args.workers)


if __name__ == "__main__":
    main()