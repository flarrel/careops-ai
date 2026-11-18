"""Build independent development subdivisions and paired synthetic scans."""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 1707
IMAGE_SIZE = 64
CLASSES = ("clean", "skewed", "occluded")
VARIANTS = 4


@dataclass(frozen=True)
class Scans:
    pixels: np.ndarray
    labels: np.ndarray
    ids: list[str]


def split_claims(rows):
    ids = [row["claim_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate development claim ID")
    by_template = defaultdict(list)
    for row in rows:
        by_template[row["template"]].append(row)
    partitions = {"train": [], "validation": [], "holdout": []}
    rng = np.random.default_rng(SEED)
    for template in sorted(by_template):
        members = sorted(by_template[template], key=lambda item: item["claim_id"])
        if len(members) < 3:
            raise ValueError("Each development template needs three or more distinct claims")
        order = rng.permutation(len(members))
        partitions["train"].extend(members[i] for i in order[:-2])
        partitions["validation"].append(members[order[-2]])
        partitions["holdout"].append(members[order[-1]])
    return partitions


def document(row, rng):
    canvas = Image.new("L", (128, 128), int(rng.integers(185, 218)))
    draw = ImageDraw.Draw(canvas)
    x, y = (int(value) for value in rng.integers(9, 19, size=2))
    width, height = (int(value) for value in rng.integers(91, 107, size=2))
    paper = (x, y, min(123, x + width), min(123, y + height))
    draw.rectangle(paper, fill=int(rng.integers(232, 256)), outline=110)
    body = ImageFont.load_default(size=int(rng.integers(6, 9)))
    title = ImageFont.load_default(size=8)
    ink = int(rng.integers(17, 62))
    draw.text((x + 5, y + 4), "SYNTHETIC CLAIM", font=title, fill=ink)
    draw.line((x + 5, y + 17, paper[2] - 5, y + 17), fill=ink)
    values = [
        row["claim_id"], row["service_type"], f"IDR {row['amount_idr']:,}",
        *[text[:19] for text in row["attachments"]],
    ]
    row_gap = int(rng.integers(9, 13))
    for index, value in enumerate(values):
        draw.text((x + 7, y + 21 + index * row_gap), value, font=body, fill=ink)
    for offset in (75, 84):
        if y + offset < paper[3] - 2:
            draw.line((x + 5, y + offset, paper[2] - 6, y + offset), fill=ink)
    return canvas, paper


def render_scan(row, variant, label):
    seed = f"{SEED}:{row['claim_id']}:{variant}".encode()
    rng = np.random.default_rng(int.from_bytes(sha256(seed).digest()[:8]))
    image, (left, top, right, bottom) = document(row, rng)
    angle = float(rng.uniform(-2.2, 2.2))
    if label == "skewed":
        angle = float(rng.choice([-1, 1]) * rng.uniform(11, 23))
    elif label == "occluded":
        x = int(rng.integers(left + 7, right - 28))
        y = int(rng.integers(top + 17, bottom - 30))
        width, height = (int(value) for value in rng.integers(20, 29, size=2))
        ImageDraw.Draw(image).rectangle((x, y, x + width, y + height), fill=111)
    elif label != "clean":
        raise ValueError(f"Unknown scan condition: {label}")
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=200)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0, 2.5, pixels.shape)
    return np.clip(pixels + noise, 0, 255).astype(np.uint8)


def make_scans(rows):
    partitions = {}
    for split, claims in split_claims(rows).items():
        pixels, labels, ids = [], [], []
        for row in claims:
            for variant in range(VARIANTS):
                for index, label in enumerate(CLASSES):
                    pixels.append(render_scan(row, variant, label))
                    labels.append(index)
                    ids.append(f"{row['claim_id']}:{variant}:{label}")
        partitions[split] = Scans(np.stack(pixels), np.asarray(labels), ids)
    return partitions
