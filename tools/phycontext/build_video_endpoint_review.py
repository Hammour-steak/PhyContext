#!/usr/bin/env python3
"""Build a labeled, fixed-frame review sheet from several videos."""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        metavar="LABEL=VIDEO",
        help="Labeled video row; repeat in desired top-to-bottom order.",
    )
    parser.add_argument("--frames", default="0,24,48,72")
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=176)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_row(value: str) -> tuple[str, Path]:
    label, separator, video = value.partition("=")
    if not separator or not label.strip() or not video.strip():
        raise ValueError(f"Invalid --row {value!r}; expected LABEL=VIDEO")
    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(path)
    return label.strip(), path


def read_frame(video: Path, frame_index: int) -> Image.Image:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    with Image.open(io.BytesIO(result.stdout)) as image:
        return image.convert("RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    rows = [parse_row(value) for value in args.row]
    frame_indices = [int(value) for value in args.frames.split(",")]
    if not frame_indices or min(frame_indices) < 0:
        raise ValueError("--frames must contain non-negative frame indices")

    sheet = Image.new(
        "RGB",
        (args.tile_width * len(frame_indices), args.tile_height * len(rows)),
        "black",
    )
    font = load_font(max(16, args.tile_height // 8))

    for row_index, (label, video) in enumerate(rows):
        y = row_index * args.tile_height
        for column, frame_index in enumerate(frame_indices):
            frame = read_frame(video, frame_index)
            frame = frame.resize((args.tile_width, args.tile_height), Image.Resampling.LANCZOS)
            sheet.paste(frame, (column * args.tile_width, y))

        draw = ImageDraw.Draw(sheet)
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            (8, y + 6, 16 + bounds[2] - bounds[0], y + 12 + bounds[3] - bounds[1]),
            fill=(0, 0, 0),
        )
        draw.text((12, y + 8), label, fill="white", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
