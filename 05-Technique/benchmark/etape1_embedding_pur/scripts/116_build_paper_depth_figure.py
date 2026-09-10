"""Build the A3/E029 four-panel depth figure from frozen, aggregate-only artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


EXPECTED_K_RETRIEVAL = set(range(1, 101))
EXPECTED_K_RERANKING = {10, 20, 30, 40, 50, 60, 70}
EXPECTED_K70 = {
    ("statutory_articles", "cosine"): 0.570,
    ("statutory_articles", "ppr"): 0.666,
    ("statutory_articles", "lightgcn"): 0.654,
    ("judicial_decisions", "cosine"): 0.298,
    ("judicial_decisions", "ppr"): 0.309,
    ("judicial_decisions", "lightgcn"): 0.329,
}
METHOD_COLORS = {"cosine": "#0072B2", "ppr": "#E69F00", "lightgcn": "#009E73"}
TARGETS = {"articles": "statutory_articles", "jurisprudence": "judicial_decisions", "article": "statutory_articles", "jp": "judicial_decisions"}
DISPLAY_TARGETS = {"statutory_articles": "Statutory Articles", "judicial_decisions": "Judicial Decisions"}
GRAPH_LABELS = {
    ("retrieval", "statutory_articles", "cosine"): "BGE-M3 cosine",
    ("retrieval", "statutory_articles", "ppr"): "PPR G6-AA",
    ("retrieval", "statutory_articles", "lightgcn"): "LightGCN G6",
    ("retrieval", "judicial_decisions", "cosine"): "BGE-M3 cosine",
    ("retrieval", "judicial_decisions", "ppr"): "PPR G7-AA",
    ("retrieval", "judicial_decisions", "lightgcn"): "LightGCN G6",
    ("reranking", "statutory_articles", "cosine"): "BGE-M3 cosine",
    ("reranking", "statutory_articles", "ppr"): "PPR G6-AA",
    ("reranking", "statutory_articles", "lightgcn"): "LightGCN G6",
    ("reranking", "judicial_decisions", "cosine"): "BGE-M3 cosine",
    ("reranking", "judicial_decisions", "ppr"): "PPR G6-AA",
    ("reranking", "judicial_decisions", "lightgcn"): "LightGCN G6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")


def _validate_depth_source(frame: pd.DataFrame, *, source: str, target: str, seeds: int, label: str) -> pd.DataFrame:
    _require_columns(frame, ("source", "target", "k", "mean", "seeds"), label)
    selected = frame.loc[(frame["source"].eq(source)) & (frame["target"].eq(target))].copy()
    if set(selected["k"].astype(int)) != EXPECTED_K_RETRIEVAL or len(selected) != 100:
        raise ValueError(f"{label}: {source}/{target} must contain exactly K=1..100")
    if set(selected["seeds"].astype(int)) != {seeds}:
        raise ValueError(f"{label}: unexpected seed count for {source}/{target}")
    return selected.sort_values("k", kind="stable")


def prepare_retrieval_tidy(r5: pd.DataFrame, lightgcn: pd.DataFrame, ppr_g6_jp: pd.DataFrame) -> pd.DataFrame:
    """Return the six A3 retrieval curves and keep the G6 JP baseline separate for E029."""
    pieces = []
    sources = [
        (r5, "cosine", "articles", 1, "r5"),
        (r5, "cosine", "jurisprudence", 1, "r5"),
        (r5, "ppr", "articles", 1, "r5"),
        (r5, "ppr", "jurisprudence", 1, "r5"),
        (lightgcn, "lightgcn_g6", "articles", 3, "lightgcn_g6"),
        (lightgcn, "lightgcn_g6", "jurisprudence", 3, "lightgcn_g6"),
    ]
    for frame, source, source_target, seeds, label in sources:
        selected = _validate_depth_source(frame, source=source, target=source_target, seeds=seeds, label=label)
        target = TARGETS[source_target]
        method = "lightgcn" if source == "lightgcn_g6" else source
        pieces.append(pd.DataFrame({
            "target": target,
            "method": method,
            "graph_label": GRAPH_LABELS[("retrieval", target, method)],
            "k": selected["k"].astype(int).to_numpy(),
            "nhit_at_k": selected["mean"].astype(float).to_numpy(),
            "seed_count": selected["seeds"].astype(int).to_numpy(),
        }))
    _validate_depth_source(
        ppr_g6_jp, source="ppr_g6_jurisprudence", target="jurisprudence", seeds=1, label="ppr_g6_jp"
    )
    result = pd.concat(pieces, ignore_index=True).sort_values(["target", "method", "k"], kind="stable")
    if len(result) != 600:
        raise ValueError(f"retrieval tidy export must have 600 rows, found {len(result)}")
    return result.reset_index(drop=True)


def _baseline_values(retrieval: pd.DataFrame, ppr_g6_jp: pd.DataFrame) -> dict[tuple[str, str], float]:
    baseline = {}
    for target in DISPLAY_TARGETS:
        for method in ("cosine", "ppr", "lightgcn"):
            if target == "judicial_decisions" and method == "ppr":
                selected = _validate_depth_source(
                    ppr_g6_jp, source="ppr_g6_jurisprudence", target="jurisprudence", seeds=1, label="ppr_g6_jp"
                )
                baseline[(target, method)] = float(selected.loc[selected["k"].eq(10), "mean"].item())
            else:
                selected = retrieval.loc[
                    (retrieval["target"].eq(target)) & (retrieval["method"].eq(method)) & (retrieval["k"].eq(10))
                ]
                if len(selected) != 1:
                    raise ValueError(f"missing retrieval baseline for {target}/{method}")
                baseline[(target, method)] = float(selected["nhit_at_k"].item())
    return baseline


def prepare_reranking_tidy(per_seed: pd.DataFrame, seed_mean: pd.DataFrame, retrieval: pd.DataFrame, ppr_g6_jp: pd.DataFrame) -> pd.DataFrame:
    """Validate seed aggregation and produce the 42 frozen E029 curve points."""
    _require_columns(per_seed, ("family", "modality", "k_in", "replay_seed", "hit_at_10"), "per-seed E029 metrics")
    _require_columns(seed_mean, ("family", "modality", "k_in", "hit_at_10", "seed_count"), "seed-mean E029 metrics")
    if set(seed_mean["k_in"].astype(int)) != EXPECTED_K_RERANKING or len(seed_mean) != 42:
        raise ValueError("seed-mean E029 metrics must contain 42 method/task/depth conditions")
    expected_seed_counts = {"cosine": 1, "ppr": 1, "lightgcn": 3}
    rows = []
    baseline = _baseline_values(retrieval, ppr_g6_jp)
    for family in ("cosine", "ppr", "lightgcn"):
        for modality in ("article", "jp"):
            target = TARGETS[modality]
            mean_part = seed_mean.loc[
                (seed_mean["family"].eq(family)) & (seed_mean["modality"].eq(modality))
            ].sort_values("k_in", kind="stable")
            seed_part = per_seed.loc[
                (per_seed["family"].eq(family)) & (per_seed["modality"].eq(modality))
            ].copy()
            if len(mean_part) != 7 or set(mean_part["k_in"].astype(int)) != EXPECTED_K_RERANKING:
                raise ValueError(f"missing E029 depth for {family}/{modality}")
            if set(mean_part["seed_count"].astype(int)) != {expected_seed_counts[family]}:
                raise ValueError(f"unexpected E029 seed count for {family}/{modality}")
            for k_in, mean_row in mean_part.set_index("k_in").iterrows():
                seed_values = seed_part.loc[seed_part["k_in"].astype(int).eq(int(k_in)), "hit_at_10"].astype(float)
                if len(seed_values) != expected_seed_counts[family]:
                    raise ValueError(f"missing E029 per-seed evidence for {family}/{modality}/K={k_in}")
                if abs(float(seed_values.mean()) - float(mean_row["hit_at_10"])) > 1e-12:
                    raise ValueError(f"seed mean mismatch for {family}/{modality}/K={k_in}")
                value = float(mean_row["hit_at_10"])
                rows.append({
                    "target": target,
                    "method": family,
                    "graph_label": GRAPH_LABELS[("reranking", target, family)],
                    "k_in": int(k_in),
                    "nhit_at_10": value,
                    "seed_count": int(mean_row["seed_count"]),
                    "retriever_nhit_at_10": baseline[(target, family)],
                    "difference_from_retriever": value - baseline[(target, family)],
                })
    result = pd.DataFrame(rows).sort_values(["target", "method", "k_in"], kind="stable").reset_index(drop=True)
    if len(result) != 42:
        raise ValueError(f"reranking tidy export must have 42 rows, found {len(result)}")
    for key, expected in EXPECTED_K70.items():
        observed = result.loc[
            (result["target"].eq(key[0])) & (result["method"].eq(key[1])) & (result["k_in"].eq(70)), "nhit_at_10"
        ]
        if len(observed) != 1 or abs(float(observed.item()) - expected) > 0.0006:
            raise ValueError(f"E029 K=70 verification failed for {key}: expected about {expected}")
    return result


def _pdf_panel(pdf: Canvas, box: tuple[float, float, float, float], frame: pd.DataFrame, *, x_col: str, y_col: str, title: str, y_label: str, y_max: float, baselines: dict[str, float] | None = None) -> None:
    x, y, width, height = box
    left, bottom, right, top = x + 40, y + 32, x + width - 10, y + height - 22
    x_min, x_max = int(frame[x_col].min()), int(frame[x_col].max())
    def px(value: float) -> float: return left + (value - x_min) / (x_max - x_min) * (right - left)
    def py(value: float) -> float: return bottom + value / y_max * (top - bottom)
    pdf.setStrokeColor(HexColor("#404040")); pdf.setLineWidth(0.5)
    pdf.line(left, bottom, right, bottom); pdf.line(left, bottom, left, top)
    pdf.setFillColor(HexColor("#202020")); pdf.setFont("Helvetica-Bold", 9); pdf.drawString(x, y + height - 12, title)
    pdf.setFont("Helvetica", 7)
    for tick in (x_min, 10, 50, x_max) if x_col == "k" else (10, 20, 40, 70):
        if x_min <= tick <= x_max:
            pdf.setStrokeColor(HexColor("#D9D9D9")); pdf.line(px(tick), bottom, px(tick), top)
            pdf.setFillColor(HexColor("#303030")); pdf.drawCentredString(px(tick), bottom - 12, str(tick))
    for tick in (0.0, y_max / 2, y_max):
        pdf.setStrokeColor(HexColor("#E6E6E6")); pdf.line(left, py(tick), right, py(tick))
        pdf.setFillColor(HexColor("#303030")); pdf.drawRightString(left - 4, py(tick) - 2, f"{tick:.1f}")
    pdf.saveState(); pdf.translate(x + 10, y + height / 2); pdf.rotate(90); pdf.drawCentredString(0, 0, y_label); pdf.restoreState()
    pdf.drawCentredString((left + right) / 2, y + 5, "K" if x_col == "k" else "K_in")
    for method, part in frame.groupby("method", sort=False):
        color = HexColor(METHOD_COLORS[method])
        if baselines is not None:
            pdf.setStrokeColor(color); pdf.setLineWidth(0.7); pdf.setDash(2, 2); pdf.line(left, py(baselines[method]), right, py(baselines[method])); pdf.setDash()
        points = list(zip(part[x_col].astype(float), part[y_col].astype(float), strict=True))
        pdf.setStrokeColor(color); pdf.setLineWidth(1.4)
        path = pdf.beginPath(); path.moveTo(px(points[0][0]), py(points[0][1]))
        for x_value, y_value in points[1:]: path.lineTo(px(x_value), py(y_value))
        pdf.drawPath(path)
        if baselines is not None:
            pdf.setFillColor(color)
            for x_value, y_value in points: pdf.circle(px(x_value), py(y_value), 1.6, fill=1, stroke=0)


def render_pdf(path: Path, retrieval: pd.DataFrame, reranking: pd.DataFrame) -> None:
    pdf = Canvas(str(path), pagesize=(792, 504))
    pdf.setTitle("A3 retrieval and E029 reranking depth")
    pdf.setFont("Helvetica-Bold", 11); pdf.drawCentredString(396, 488, "Retrieval and reranking depth")
    pdf.setFont("Helvetica", 7)
    for index, method in enumerate(("cosine", "ppr", "lightgcn")):
        x = 195 + index * 108; pdf.setStrokeColor(HexColor(METHOD_COLORS[method])); pdf.setLineWidth(1.6); pdf.line(x, 470, x + 14, 470)
        pdf.setFillColor(HexColor("#303030")); pdf.drawString(x + 18, 467, {"cosine": "BGE-M3 cosine", "ppr": "PPR", "lightgcn": "LightGCN G6"}[method])
    pdf.setStrokeColor(HexColor("#606060")); pdf.setDash(2, 2); pdf.line(546, 470, 560, 470); pdf.setDash(); pdf.setFillColor(HexColor("#303030")); pdf.drawString(564, 467, "retriever @10")
    boxes = [(28, 251, 360, 205), (404, 251, 360, 205), (28, 30, 360, 205), (404, 30, 360, 205)]
    for box, panel, target, mode in zip(boxes, ("A", "B", "C", "D"), ("statutory_articles", "judicial_decisions", "statutory_articles", "judicial_decisions"), ("retrieval", "retrieval", "reranking", "reranking"), strict=True):
        if mode == "retrieval":
            _pdf_panel(pdf, box, retrieval.loc[retrieval.target.eq(target)], x_col="k", y_col="nhit_at_k", title=f"{panel}. {DISPLAY_TARGETS[target]} — Retrieval depth", y_label="NHit@K", y_max=0.8 if target == "statutory_articles" else 0.4)
        else:
            part = reranking.loc[reranking.target.eq(target)]
            _pdf_panel(pdf, box, part, x_col="k_in", y_col="nhit_at_10", title=f"{panel}. {DISPLAY_TARGETS[target]} — Reranking depth", y_label="Final NHit@10", y_max=0.8 if target == "statutory_articles" else 0.4, baselines=dict(part.groupby("method")["retriever_nhit_at_10"].first()))
    pdf.save()


def render_png(path: Path, retrieval: pd.DataFrame, reranking: pd.DataFrame) -> None:
    image = Image.new("RGB", (2400, 1500), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana.ttf", 19)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana.ttf", 15)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Verdana Bold.ttf", 23)
    title = "Retrieval and reranking depth"
    title_width = draw.textbbox((0, 0), title, font=bold)[2]
    draw.text(((2400 - title_width) / 2, 20), title, fill="#202020", font=bold)
    legend_x = 710
    for index, method in enumerate(("cosine", "ppr", "lightgcn")):
        x = legend_x + index * 250
        draw.line((x, 62, x + 32, 62), fill=METHOD_COLORS[method], width=4)
        label = {"cosine": "BGE-M3 cosine", "ppr": "PPR", "lightgcn": "LightGCN G6"}[method]
        draw.text((x + 42, 50), label, fill="#303030", font=small)
    draw.line((1455, 62, 1487, 62), fill="#606060", width=2)
    for start in range(1455, 1487, 8):
        draw.line((start, 62, min(start + 4, 1487), 62), fill="white", width=2)
    draw.text((1498, 50), "retriever @10", fill="#303030", font=small)
    boxes = [(70, 100, 1110, 720), (1290, 100, 2330, 720), (70, 800, 1110, 1420), (1290, 800, 2330, 1420)]
    for box, panel, target, mode in zip(boxes, ("A", "B", "C", "D"), ("statutory_articles", "judicial_decisions", "statutory_articles", "judicial_decisions"), ("retrieval", "retrieval", "reranking", "reranking"), strict=True):
        x, y, right, bottom = box; left, top = x + 100, y + 60; width, height = right - left - 30, bottom - top - 70
        frame = retrieval.loc[retrieval.target.eq(target)] if mode == "retrieval" else reranking.loc[reranking.target.eq(target)]
        x_col, y_col = ("k", "nhit_at_k") if mode == "retrieval" else ("k_in", "nhit_at_10")
        y_max = 0.8 if target == "statutory_articles" else 0.4; x_min, x_max = int(frame[x_col].min()), int(frame[x_col].max())
        px = lambda value: left + (value - x_min) / (x_max - x_min) * width
        py = lambda value: top + height - value / y_max * height
        draw.text((x, y), f"{panel}. {DISPLAY_TARGETS[target]} — {'Retrieval depth' if mode == 'retrieval' else 'Reranking depth'}", fill="#202020", font=regular)
        draw.line((left, top, left, top + height), fill="#404040", width=2); draw.line((left, top + height, left + width, top + height), fill="#404040", width=2)
        x_ticks = (x_min, 10, 50, x_max) if mode == "retrieval" else (10, 20, 40, 70)
        for tick in x_ticks:
            if x_min <= tick <= x_max:
                xx = px(tick)
                draw.line((xx, top, xx, top + height), fill="#D9D9D9", width=1)
                label = str(tick)
                label_width = draw.textbbox((0, 0), label, font=small)[2]
                draw.text((xx - label_width / 2, top + height + 10), label, fill="#303030", font=small)
        for tick in (0.0, y_max / 2, y_max):
            yy = py(tick)
            draw.line((left, yy, left + width, yy), fill="#E6E6E6", width=1)
            label = f"{tick:.1f}"
            label_width = draw.textbbox((0, 0), label, font=small)[2]
            draw.text((left - label_width - 10, yy - 9), label, fill="#303030", font=small)
        x_label = "K" if mode == "retrieval" else "K_in"
        x_label_width = draw.textbbox((0, 0), x_label, font=small)[2]
        draw.text((left + (width - x_label_width) / 2, top + height + 38), x_label, fill="#303030", font=small)
        y_label = "NHit@K" if mode == "retrieval" else "Final NHit@10"
        draw.text((x, top + height / 2 - 8), y_label, fill="#303030", font=small)
        for method, part in frame.groupby("method", sort=False):
            color = METHOD_COLORS[method]
            if mode == "reranking":
                baseline = float(part["retriever_nhit_at_10"].iloc[0]); yy = py(baseline)
                for start in range(int(left), int(left + width), 16): draw.line((start, yy, min(start + 8, left + width), yy), fill=color, width=1)
            points = [(px(float(row[x_col])), py(float(row[y_col]))) for _, row in part.iterrows()]
            draw.line(points, fill=color, width=4)
            if mode == "reranking":
                for point in points: draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    image.save(path, dpi=(320, 320))


def build(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "retrieval_r5": args.retrieval_r5,
        "retrieval_lightgcn": args.retrieval_lightgcn,
        "retrieval_ppr_g6_jp": args.retrieval_ppr_g6_jp,
        "reranking_per_seed": args.reranking_per_seed,
        "reranking_seed_mean": args.reranking_seed_mean,
    }
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    if args.provenance is not None and not args.provenance.is_file():
        raise FileNotFoundError(args.provenance)
    retrieval = prepare_retrieval_tidy(pd.read_csv(args.retrieval_r5), pd.read_csv(args.retrieval_lightgcn), pd.read_csv(args.retrieval_ppr_g6_jp))
    reranking = prepare_reranking_tidy(pd.read_csv(args.reranking_per_seed), pd.read_csv(args.reranking_seed_mean), retrieval, pd.read_csv(args.retrieval_ppr_g6_jp))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    retrieval_path = args.out_dir / "retrieval_depth_tidy.csv"; reranking_path = args.out_dir / "reranking_depth_tidy.csv"
    pdf_path = args.out_dir / "paper_depth_figure.pdf"; png_path = args.out_dir / "paper_depth_figure.png"; readme_path = args.out_dir / "README.md"; manifest_path = args.out_dir / "depth_figure_manifest.json"
    retrieval.to_csv(retrieval_path, index=False); reranking.to_csv(reranking_path, index=False)
    render_pdf(pdf_path, retrieval, reranking); render_png(png_path, retrieval, reranking)
    source_contract = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance is not None else None
    caption = ("Retrieval and reranking depth on the frozen A3 internal evaluation (754 questions). Top panels report exact NHit@K from frozen top-100 rankings. Bottom panels report final NHit@10 after E029 reranking at K_out=10; solid curves are reranked results and dashed lines are the corresponding retriever NHit@10. The jurisprudence PPR reranking curve uses G6-AA frozen pools and is distinct from the PPR G7-AA retrieval line in the main table. E029 is exploratory.")
    readme = "\n".join([
        "# A3/E029 depth figure",
        "",
        caption,
        "",
        "## Scope",
        "",
        "- Evaluation: frozen A3 internal evaluation, 754 questions.",
        "- Retrieval panels: exact NHit@K for K=1..100 from frozen top-100 rankings.",
        "- Reranking panels: E029 output-512, K_in=10,20,30,40,50,60,70 and K_out=10.",
        "- LightGCN: mean of frozen replay seeds 42, 43 and 44; seed-level E029 inputs are verified before aggregation.",
        "- Articles: BGE-M3 cosine, PPR G6-AA and LightGCN G6. Judicial Decisions retrieval: BGE-M3 cosine, PPR G7-AA and LightGCN G6.",
        "- Judicial Decisions reranking: BGE-M3 cosine, PPR G6-AA and LightGCN G6. The PPR G6-AA baseline is 0.22822723253757737; this is not the PPR G7-AA main-table row.",
        "",
        "## Provenance and status",
        "",
        f"- Canonical source contract: {args.provenance} (SHA-256 `{sha256(args.provenance) if args.provenance is not None else 'not supplied'}`).",
        "- All values are derived from frozen, hash-verified aggregate artifacts. No model, retrieval, training, selection, or E030 job is run by this script.",
        "- Retrieval is A3 internal evaluation after train/CV freeze. E029 reranking is exploratory and must not support a confirmatory superiority claim.",
        "",
    ])
    readme_path.write_text(readme, encoding="utf-8")
    manifest = {
        "schema_version": "paper-depth-figure.v1",
        "a3_evaluation_questions": args.expected_questions,
        "coverage": {"questions": args.expected_questions},
        "retrieval_coverage": {"k": "1..100", "rows": len(retrieval), "methods_per_task": 3},
        "reranking_coverage": {"k_in": [10, 20, 30, 40, 50, 60, 70], "rows": len(reranking), "k_out": 10, "lightgcn_seed_policy": "mean of frozen seeds 42,43,44"},
        "figure_layout": {"panels": ["A", "B", "C", "D"], "colors": {"cosine": "#0072B2", "ppr": "#E69F00", "lightgcn": "#009E73"}, "line_semantics": {"solid": "reranked top-10", "dashed": "retriever NHit@10"}},
        "scientific_status": {"retrieval": "A3 internal evaluation after train/CV freeze", "reranking": "exploratory E029 annex"},
        "ppr_jp_reranking_scope": "PPR G6-AA frozen pools; not the PPR G7-AA main-table retrieval configuration",
        "caption": caption,
        "provenance": source_contract,
        "canonical_source_contract": source_contract,
        "provenance_sha256": sha256(args.provenance) if args.provenance is not None else None,
        "staged_input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "outputs_sha256": {},
    }
    manifest["outputs_sha256"] = {name: sha256(path) for name, path in {"retrieval_depth_tidy.csv": retrieval_path, "reranking_depth_tidy.csv": reranking_path, "paper_depth_figure.pdf": pdf_path, "paper_depth_figure.png": png_path, "README.md": readme_path}.items()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"retrieval": retrieval_path, "reranking": reranking_path, "pdf": pdf_path, "png": png_path, "readme": readme_path, "manifest": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-r5", type=Path, required=True)
    parser.add_argument("--retrieval-lightgcn", type=Path, required=True)
    parser.add_argument("--retrieval-ppr-g6-jp", type=Path, required=True)
    parser.add_argument("--reranking-per-seed", type=Path, required=True)
    parser.add_argument("--reranking-seed-mean", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=754)
    args = parser.parse_args()
    print(json.dumps({key: str(value) for key, value in build(args).items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
