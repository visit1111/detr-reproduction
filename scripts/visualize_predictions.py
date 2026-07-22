import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hubconf import detr_resnet50
from util.box_ops import box_cxcywh_to_xyxy


THEMES = [
    ("People and bicycle", {"person", "bicycle"}),
    ("Street traffic", {"car", "traffic light"}),
    ("Dog", {"dog"}),
    ("Elephant", {"elephant"}),
    ("Zebra", {"zebra"}),
    ("Food", {"pizza"}),
]

PALETTE = [
    "#ff5c7a", "#00a6a6", "#ff9f1c", "#6c63ff", "#2f80ed",
    "#27ae60", "#d35400", "#9b51e0", "#eb5757", "#0081a7",
]


def load_font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def build_candidate_groups(annotation_path: Path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            by_image[ann["image_id"]].append(ann)

    groups = []
    for title, required_names in THEMES:
        required_ids = {cid for cid, name in categories.items() if name in required_names}
        candidates = []
        for image_id, anns in by_image.items():
            present = {ann["category_id"] for ann in anns}
            if not required_ids.issubset(present):
                continue
            im = images[image_id]
            image_area = im["width"] * im["height"]
            target_area = sum(
                ann["area"] for ann in anns if ann["category_id"] in required_ids
            )
            target_count = sum(ann["category_id"] in required_ids for ann in anns)
            score = target_area / image_area + min(target_count, 4) * 0.02
            candidates.append((score, image_id))
        if not candidates:
            raise RuntimeError(f"No COCO image found for theme: {title}")
        candidates.sort(reverse=True)
        groups.append((title, required_ids, candidates[:30]))
    return groups, categories, images, by_image


def make_transform():
    return T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def scale_boxes(boxes, image_size):
    width, height = image_size
    boxes = box_cxcywh_to_xyxy(boxes)
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=boxes.device)
    return boxes * scale


def infer(model, image, device, threshold):
    tensor = make_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
    probabilities = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    scores, labels = probabilities.max(-1)
    keep = scores >= threshold
    boxes = scale_boxes(outputs["pred_boxes"][0, keep], image.size)
    predictions = []
    for score, label, box in zip(scores[keep], labels[keep], boxes):
        predictions.append({
            "score": float(score.cpu()),
            "category_id": int(label.cpu()),
            "box": [float(v) for v in box.cpu()],
        })
    return sorted(predictions, key=lambda x: x["score"], reverse=True)


def draw_label(draw, xy, text, color, font):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font, stroke_width=0)
    padding = 4
    draw.rectangle(
        (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
        fill=color,
    )
    draw.text((x, y), text, fill="white", font=font)


def draw_predictions(image, predictions, categories):
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    width = max(3, round(min(result.size) / 180))
    font = load_font(max(14, round(min(result.size) / 38)), bold=True)
    for pred in predictions[:25]:
        category_id = pred["category_id"]
        color = PALETTE[category_id % len(PALETTE)]
        x0, y0, x1, y1 = pred["box"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        name = categories.get(category_id, f"class {category_id}")
        label_y = max(2, y0 - font.size - 8)
        draw_label(draw, (x0 + 3, label_y), f"{name} {pred['score']:.2f}", color, font)
    return result


def draw_ground_truth(image, annotations, categories):
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    color = "#1677ff"
    width = max(3, round(min(result.size) / 180))
    font = load_font(max(14, round(min(result.size) / 38)), bold=True)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=width)
        label_y = max(2, y - font.size - 8)
        draw_label(draw, (x + 3, label_y), categories[ann["category_id"]], color, font)
    return result


def panel(image, title, size=(760, 520)):
    canvas = Image.new("RGB", size, "#f3f5f7")
    header = 46
    contained = ImageOps.contain(image, (size[0] - 20, size[1] - header - 16), Image.Resampling.LANCZOS)
    x = (size[0] - contained.width) // 2
    y = header + (size[1] - header - contained.height) // 2
    canvas.paste(contained, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 11), title, fill="#20242a", font=load_font(21, bold=True))
    return canvas


def make_grid(panels, columns, gap=16):
    rows = (len(panels) + columns - 1) // columns
    width = columns * panels[0].width + (columns - 1) * gap
    height = rows * panels[0].height + (rows - 1) * gap
    grid = Image.new("RGB", (width, height), "#dfe3e8")
    for index, item in enumerate(panels):
        x = (index % columns) * (item.width + gap)
        y = (index // columns) * (item.height + gap)
        grid.paste(item, (x, y))
    return grid


def make_metrics_chart(output_path: Path):
    labels = ["AP", "AP50", "AP75", "AP small", "AP medium", "AP large"]
    values = [42.0, 62.4, 44.2, 20.5, 45.8, 61.1]
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    bars = ax.barh(labels[::-1], values[::-1], color="#1677ff")
    ax.set_xlim(0, 70)
    ax.set_xlabel("COCO score (%)")
    ax.set_title("DETR-R50 on COCO val2017", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, categories, images, by_image = build_candidate_groups(
        args.coco_path / "annotations" / "instances_val2017.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = detr_resnet50(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    selected = []
    used_image_ids = set()
    for theme, required_ids, candidates in groups:
        fallback = None
        chosen = None
        for _, image_id in candidates:
            if image_id in used_image_ids:
                continue
            image_info = images[image_id]
            image_path = args.coco_path / "val2017" / image_info["file_name"]
            image = Image.open(image_path).convert("RGB")
            predictions = infer(model, image, device, args.threshold)
            option = (theme, image_info, by_image[image_id], image, predictions)
            if fallback is None:
                fallback = option
            predicted_ids = {pred["category_id"] for pred in predictions}
            if required_ids.issubset(predicted_ids):
                chosen = option
                break
        if chosen is None:
            chosen = fallback
        if chosen is None:
            raise RuntimeError(f"No unused COCO candidate available for theme: {theme}")
        used_image_ids.add(chosen[1]["id"])
        selected.append(chosen)

    prediction_panels = []
    comparison_panels = []
    records = []
    for index, (theme, image_info, annotations, image, predictions) in enumerate(selected, start=1):
        predicted_image = draw_predictions(image, predictions, categories)
        ground_truth_image = draw_ground_truth(image, annotations, categories)

        filename = f"sample-{index:02d}-{theme.lower().replace(' ', '-').replace('&', 'and')}.jpg"
        predicted_image.save(args.output_dir / filename, quality=95)
        prediction_panels.append(panel(predicted_image, f"{theme} · DETR predictions"))
        if index <= 4:
            comparison_panels.append(panel(ground_truth_image, f"{theme} · Ground truth"))
            comparison_panels.append(panel(predicted_image, f"{theme} · DETR predictions"))

        records.append({
            "theme": theme,
            "image_id": image_info["id"],
            "file_name": image_info["file_name"],
            "prediction_count": len(predictions),
            "predictions": predictions,
        })

    make_grid(prediction_panels, columns=2).save(
        args.output_dir / "detr-predictions-grid.jpg", quality=95
    )
    make_grid(comparison_panels, columns=2).save(
        args.output_dir / "ground-truth-vs-detr.jpg", quality=95
    )
    make_metrics_chart(args.output_dir / "coco-metrics.png")
    (args.output_dir / "selected-samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "device": str(device),
        "samples": len(records),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
