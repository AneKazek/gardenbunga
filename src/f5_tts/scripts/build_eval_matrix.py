import argparse
import csv
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Build evaluation matrix for baseline vs hybrid.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="baseline_f5")
    parser.add_argument("--hybrid-name", default="hybrid_f5_mamba")
    parser.add_argument("--smos-ratings", default=None)
    parser.add_argument("--cmos-ratings", default=None)
    parser.add_argument("--cmos-key", default=None)
    return parser.parse_args()


def read_jsonl_metric(path: Path, metric_key: str):
    if not path.exists():
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        payload = json.loads(line)
        if metric_key in payload:
            values.append(float(payload[metric_key]))
    if not values:
        return None
    return sum(values) / len(values)


def read_smos_scores(path: Path | None):
    if path is None or not path.exists():
        return {}
    scores = {}
    counts = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row.get("model", "").strip()
            score = row.get("smos_score", "").strip()
            if not model or not score:
                continue
            scores[model] = scores.get(model, 0.0) + float(score)
            counts[model] = counts.get(model, 0) + 1
    return {model: scores[model] / counts[model] for model in scores}


def read_cmos_scores(path: Path | None, key_path: Path | None):
    if path is None or key_path is None or not path.exists() or not key_path.exists():
        return {}

    key_by_sample = {}
    with open(key_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key_by_sample[row["sample_id"]] = row

    totals = {}
    counts = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            score = row.get("cmos_score", "").strip()
            if not sample_id or not score or sample_id not in key_by_sample:
                continue
            score = float(score)
            key = key_by_sample[sample_id]
            model_a = key["system_a_model"]
            model_b = key["system_b_model"]
            totals[model_a] = totals.get(model_a, 0.0) + score
            counts[model_a] = counts.get(model_a, 0) + 1
            totals[model_b] = totals.get(model_b, 0.0) - score
            counts[model_b] = counts.get(model_b, 0) + 1

    return {model: totals[model] / counts[model] for model in totals}


def build_row(model_name: str, model_dir: Path, smos_scores: dict, cmos_scores: dict):
    return {
        "model": model_name,
        "wer": read_jsonl_metric(model_dir / "_wer_results.jsonl", "wer"),
        "sim_o": read_jsonl_metric(model_dir / "_sim_results.jsonl", "sim"),
        "smos": smos_scores.get(model_name),
        "cmos": cmos_scores.get(model_name),
    }


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smos_scores = read_smos_scores(Path(args.smos_ratings) if args.smos_ratings else None)
    cmos_scores = read_cmos_scores(
        Path(args.cmos_ratings) if args.cmos_ratings else None,
        Path(args.cmos_key) if args.cmos_key else None,
    )

    rows = [
        build_row(args.baseline_name, Path(args.baseline_dir), smos_scores, cmos_scores),
        build_row(args.hybrid_name, Path(args.hybrid_dir), smos_scores, cmos_scores),
    ]

    csv_path = output_dir / "evaluation_matrix.csv"
    json_path = output_dir / "evaluation_matrix.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
