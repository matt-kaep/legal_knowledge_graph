"""Render a two-panel retrieval-only successor from a frozen tidy CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


COLORS = {"cosine": "#0072B2", "ppr": "#E69F00", "lightgcn": "#009E73"}
TARGETS = ("statutory_articles", "judicial_decisions")
DISPLAY = {"statutory_articles": "Statutory Articles", "judicial_decisions": "Judicial Decisions"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(frame: pd.DataFrame) -> None:
    required = {"target", "method", "graph_label", "k", "nhit_at_k", "seed_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"retrieval CSV is missing columns {sorted(missing)}")
    if len(frame) != 600:
        raise ValueError(f"retrieval CSV must contain 600 rows, found {len(frame)}")
    if set(frame["target"]) != set(TARGETS) or set(frame["method"]) != set(COLORS):
        raise ValueError("retrieval CSV must contain two tasks and cosine/PPR/LightGCN")
    for _, part in frame.groupby(["target", "method"]):
        if len(part) != 100 or set(part["k"].astype(int)) != set(range(1, 101)):
            raise ValueError("each retrieval curve must contain exactly K=1..100")


def y_limits(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    result = {}
    for target, part in frame.groupby("target"):
        maximum = float(part["nhit_at_k"].max())
        result[target] = (0.0, math.ceil(maximum * 1.05 / 0.05) * 0.05)
    return result


def pdf_panel(pdf: Canvas, *, box: tuple[float, float, float, float], frame: pd.DataFrame, title: str, y_max: float) -> None:
    x, y, width, height = box
    left, bottom, right, top = x + 43, y + 35, x + width - 12, y + height - 25
    px = lambda value: left + (value - 1) / 99 * (right - left)
    py = lambda value: bottom + value / y_max * (top - bottom)
    pdf.setFillColor(HexColor("#202020")); pdf.setFont("Helvetica-Bold", 10); pdf.drawString(x, y + height - 14, title)
    pdf.setStrokeColor(HexColor("#404040")); pdf.setLineWidth(0.5); pdf.line(left, bottom, right, bottom); pdf.line(left, bottom, left, top)
    pdf.setFont("Helvetica", 7)
    for tick in (1, 10, 50, 100):
        pdf.setStrokeColor(HexColor("#D9D9D9")); pdf.line(px(tick), bottom, px(tick), top)
        pdf.setFillColor(HexColor("#303030")); pdf.drawCentredString(px(tick), bottom - 12, str(tick))
    for tick in (0.0, y_max / 2, y_max):
        pdf.setStrokeColor(HexColor("#E6E6E6")); pdf.line(left, py(tick), right, py(tick))
        pdf.setFillColor(HexColor("#303030")); pdf.drawRightString(left - 4, py(tick) - 2, f"{tick:.2f}")
    pdf.drawCentredString((left + right) / 2, y + 6, "K")
    pdf.saveState(); pdf.translate(x + 11, y + height / 2); pdf.rotate(90); pdf.drawCentredString(0, 0, "NHit@K"); pdf.restoreState()
    for method, part in frame.groupby("method", sort=False):
        ordered = part.sort_values("k", kind="stable")
        points = list(zip(ordered["k"].astype(float), ordered["nhit_at_k"].astype(float), strict=True))
        pdf.setStrokeColor(HexColor(COLORS[method])); pdf.setLineWidth(1.6)
        path = pdf.beginPath(); path.moveTo(px(points[0][0]), py(points[0][1]))
        for x_value, y_value in points[1:]:
            path.lineTo(px(x_value), py(y_value))
        pdf.drawPath(path)


def render_pdf(path: Path, frame: pd.DataFrame, limits: dict[str, tuple[float, float]]) -> None:
    pdf = Canvas(str(path), pagesize=(792, 330))
    pdf.setTitle("A3 retrieval depth")
    pdf.setFont("Helvetica-Bold", 12); pdf.drawCentredString(396, 312, "Retrieval depth")
    pdf.setFont("Helvetica", 7)
    for index, method in enumerate(("cosine", "ppr", "lightgcn")):
        x = 245 + index * 130
        pdf.setStrokeColor(HexColor(COLORS[method])); pdf.setLineWidth(1.8); pdf.line(x, 294, x + 17, 294)
        pdf.setFillColor(HexColor("#303030")); pdf.drawString(x + 22, 291, {"cosine": "BGE-M3 cosine", "ppr": "PPR", "lightgcn": "LightGCN G6"}[method])
    for box, panel, target in zip(((28, 35, 360, 235), (404, 35, 360, 235)), ("A", "B"), TARGETS, strict=True):
        pdf_panel(pdf, box=box, frame=frame.loc[frame.target.eq(target)], title=f"{panel}. {DISPLAY[target]} - Retrieval depth", y_max=limits[target][1])
    pdf.save()


def render_png(path: Path, frame: pd.DataFrame, limits: dict[str, tuple[float, float]]) -> None:
    image = Image.new("RGB", (2400, 900), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana.ttf", 19)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana.ttf", 15)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana Bold.ttf", 24)
    draw.text((1045, 19), "Retrieval depth", fill="#202020", font=bold)
    for index, method in enumerate(("cosine", "ppr", "lightgcn")):
        x = 735 + index * 285
        draw.line((x, 65, x + 34, 65), fill=COLORS[method], width=4)
        draw.text((x + 45, 53), {"cosine": "BGE-M3 cosine", "ppr": "PPR", "lightgcn": "LightGCN G6"}[method], fill="#303030", font=small)
    for box, panel, target in zip(((70, 105, 1110, 845), (1290, 105, 2330, 845)), ("A", "B"), TARGETS, strict=True):
        x, y, right, bottom = box
        left, top = x + 100, y + 60
        width, height = right - left - 30, bottom - top - 75
        y_max = limits[target][1]
        px = lambda value: left + (value - 1) / 99 * width
        py = lambda value: top + height - value / y_max * height
        draw.text((x, y), f"{panel}. {DISPLAY[target]} - Retrieval depth", fill="#202020", font=regular)
        draw.line((left, top, left, top + height), fill="#404040", width=2); draw.line((left, top + height, left + width, top + height), fill="#404040", width=2)
        for tick in (1, 10, 50, 100):
            xx = px(tick); draw.line((xx, top, xx, top + height), fill="#D9D9D9", width=1)
            label = str(tick); label_width = draw.textbbox((0, 0), label, font=small)[2]
            draw.text((xx - label_width / 2, top + height + 10), label, fill="#303030", font=small)
        for tick in (0.0, y_max / 2, y_max):
            yy = py(tick); draw.line((left, yy, left + width, yy), fill="#E6E6E6", width=1)
            label = f"{tick:.2f}"; label_width = draw.textbbox((0, 0), label, font=small)[2]
            draw.text((left - label_width - 10, yy - 8), label, fill="#303030", font=small)
        draw.text((left + width / 2 - 6, top + height + 38), "K", fill="#303030", font=small)
        draw.text((x, top + height / 2 - 8), "NHit@K", fill="#303030", font=small)
        for method, part in frame.loc[frame.target.eq(target)].groupby("method", sort=False):
            ordered = part.sort_values("k", kind="stable")
            points = [(px(float(row.k)), py(float(row.nhit_at_k))) for row in ordered.itertuples(index=False)]
            draw.line(points, fill=COLORS[method], width=4)
    image.save(path, dpi=(320, 320))


def build(retrieval_csv: Path, out_dir: Path) -> dict[str, Path]:
    frame = pd.read_csv(retrieval_csv)
    validate(frame)
    limits = y_limits(frame)
    out_dir.mkdir(parents=True, exist_ok=False)
    tidy_path = out_dir / "retrieval_depth_tidy.csv"
    pdf_path = out_dir / "paper_retrieval_figure.pdf"
    png_path = out_dir / "paper_retrieval_figure.png"
    manifest_path = out_dir / "retrieval_figure_manifest.json"
    frame.to_csv(tidy_path, index=False)
    render_pdf(pdf_path, frame, limits)
    render_png(png_path, frame, limits)
    manifest = {
        "schema_version": "paper-retrieval-figure.v1",
        "layout": {"panel_count": 2, "panels": ["A", "B"], "reranking_panels": False},
        "coverage": {"rows": len(frame), "k": "1..100", "questions": 754, "methods_per_task": 3},
        "source": {"path": str(retrieval_csv), "sha256": sha256(retrieval_csv)},
        "y_limits": {target: list(value) for target, value in limits.items()},
        "outputs_sha256": {"retrieval_depth_tidy.csv": sha256(tidy_path), "paper_retrieval_figure.pdf": sha256(pdf_path), "paper_retrieval_figure.png": sha256(png_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"tidy": tidy_path, "pdf": pdf_path, "png": png_path, "manifest": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({key: str(value) for key, value in build(args.retrieval_csv, args.out_dir).items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
