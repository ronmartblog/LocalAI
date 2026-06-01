# LocalAI Studio created by Ron Martinsen March 2026 - ron@martinsen.com - Apache 2.0 License
"""Generate ≤240×240 JPEG thumbnails for Model-Guide.html sample images.

Source images: D:\\LocalAI_Doc_Sample_Replacements_2026-05-19_0935\\images\\modeldemo-*.png
Output:        C:\\LocalAI\\docs\\images\\model_demos\\modeldemo-*.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"D:\LocalAI_Doc_Sample_Replacements_2026-05-19_0935\images")
DST = ROOT / "docs" / "images" / "model_demos"

MAX_DIM = 240
JPEG_QUALITY = 82


def main() -> int:
    if not SRC.exists():
        print(f"Source dir missing: {SRC}")
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    count = 0
    for src in sorted(SRC.glob("modeldemo-*.png")):
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
            out = DST / (src.stem + ".jpg")
            im.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        size = out.stat().st_size
        total_bytes += size
        count += 1
        print(f"  {src.name} -> {out.name} ({im.size[0]}x{im.size[1]}, {size/1024:.1f} KB)")
    print(f"\nWrote {count} thumbnails to {DST} (total {total_bytes/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
