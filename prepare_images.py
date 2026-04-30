# prepare_images.py
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def prepare_one(
    in_path: Path,
    out_dir: Path,
    size: int,
    out_format: str,
    jpg_quality: int,
    keep_exif: bool,
    mode: str,
    overwrite: bool,
) -> tuple[bool, str]:
    """
    Returns: (ok, message)
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        
        stem = in_path.stem
        ext  = ".png" if out_format == "png" else ".jpg"
        out_path = out_dir / f"{stem}{ext}"

        if out_path.exists() and not overwrite:
            return True, f"skip: {out_path.name}"

        with Image.open(in_path) as img:
            img = ImageOps.exif_transpose(img)  
            img = img.convert("RGB")

            if mode == "crop":
                img = center_crop_to_square(img)
                img = img.resize((size, size), resample=Image.Resampling.LANCZOS)
            elif mode == "fit":
                
                img = ImageOps.contain(img, (size, size), method=Image.Resampling.LANCZOS)
                bg  = Image.new("RGB", (size, size), (0, 0, 0))
                x   = (size - img.size[0]) // 2
                y   = (size - img.size[1]) // 2
                bg.paste(img, (x, y))
                img = bg
            else:
                raise ValueError("mode must be 'crop' or 'fit'")

            exif_bytes = None
            if keep_exif:
                try:
                    exif_bytes = img.getexif().tobytes()
                except Exception:
                    exif_bytes = None

            if out_format == "png":
                
                img.save(out_path, format="PNG")
            else:
                save_kwargs = {
                    "format": "JPEG",
                    "quality": int(jpg_quality),
                    "subsampling": 0,
                    "optimize": True,
                }
                if exif_bytes:
                    save_kwargs["exif"] = exif_bytes
                img.save(out_path, **save_kwargs)

        return True, f"ok: {out_path.name}"

    except Exception as e:
        return False, f"fail: {in_path.name} -> {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare images: resize to NxN.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder (e.g. dataset\\real_raw)")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder (e.g. dataset\\real_512)")
    ap.add_argument("--size", type=int, default=512, help="Target size (default: 512)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png", help="Output format (default: png)")
    ap.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality if --format jpg (default: 95)")
    ap.add_argument("--keep-exif", action="store_true", help="Try to keep EXIF (works best with JPG)")
    ap.add_argument("--mode", choices=["crop", "fit"], default="crop", help="crop=центр-кроп до квадрата; fit=вписать")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        print(f"[ERROR] Input folder not found: {in_dir}")
        return

    files = list_images(in_dir)
    print(f"Found {len(files)} images in {in_dir}")
    print(f"Output: {out_dir}  size={args.size}  format={args.format}  mode={args.mode}  workers={args.workers}\n")

    ok_cnt = 0
    fail_cnt = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                prepare_one,
                p, out_dir, args.size, args.format, args.jpg_quality,
                args.keep_exif, args.mode, args.overwrite
            )
            for p in files
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            ok, msg = fut.result()
            if ok:
                ok_cnt += 1
            else:
                fail_cnt += 1
                print(msg)
            if i % 200 == 0 or i == len(futures):
                print(f"Progress: {i}/{len(futures)}  ok={ok_cnt}  fail={fail_cnt}")

    print(f"\nDone. ok={ok_cnt}, fail={fail_cnt}")
    print(f"Out dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()