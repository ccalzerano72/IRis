"""
4x Classical Upsampling Script

Upscales all images in a directory by 4x using Lanczos resampling.
Output is saved as lossless PNG.

Usage:
    python script/utils/upsample_4x.py --input_dir /path/to/images --output_dir /path/to/output
"""

import argparse
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def upsample_image(input_path: Path, output_path: Path, scale: int = 4) -> None:
    """Upscale a single image by the given scale factor using Lanczos resampling."""
    img = Image.open(input_path)
    new_size = (img.width * scale, img.height * scale)
    img_upscaled = img.resize(new_size, resample=Image.LANCZOS)
    img_upscaled.save(output_path, format="PNG")


def main():
    parser = argparse.ArgumentParser(
        description="4x classical upsampling using Lanczos resampling."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where upscaled images will be saved.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscaling factor (default: 4).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all supported image files
    image_files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Upscaling {args.scale}x with Lanczos...")

    for i, img_path in enumerate(image_files, start=1):
        output_path = output_dir / f"{img_path.stem}.png"
        upsample_image(img_path, output_path, scale=args.scale)
        print(f"  [{i}/{len(image_files)}] {img_path.name} -> {output_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
