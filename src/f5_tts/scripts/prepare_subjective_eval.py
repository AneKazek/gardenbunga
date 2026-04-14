import argparse
import csv
import random
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from f5_tts.eval.utils_eval import get_librispeech_test_clean_metainfo, get_seedtts_testset_metainfo


def get_args():
    parser = argparse.ArgumentParser(description="Prepare SMOS/CMOS subjective evaluation manifests.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--task", choices=["seedtts", "librispeech"], required=True)
    parser.add_argument("--meta-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--librispeech-test-clean-path", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--baseline-name", default="baseline_f5")
    parser.add_argument("--hybrid-name", default="hybrid_f5_mamba")
    return parser.parse_args()


def load_metainfo(args):
    if args.task == "seedtts":
        return get_seedtts_testset_metainfo(args.meta_file)
    if args.librispeech_test_clean_path is None:
        raise ValueError("--librispeech-test-clean-path is required for librispeech task.")
    return get_librispeech_test_clean_metainfo(args.meta_file, args.librispeech_test_clean_path)


def main():
    args = get_args()
    baseline_dir = Path(args.baseline_dir)
    hybrid_dir = Path(args.hybrid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smos_manifest = output_dir / "smos_manifest.csv"
    cmos_blind_manifest = output_dir / "cmos_blind_manifest.csv"
    cmos_key = output_dir / "cmos_key.csv"

    metainfo = load_metainfo(args)
    rng = random.Random(args.seed)

    smos_rows = []
    cmos_rows = []
    key_rows = []

    for row in metainfo:
        utt, prompt_text, prompt_wav, gt_text, gt_wav = row
        baseline_wav = baseline_dir / f"{utt}.wav"
        hybrid_wav = hybrid_dir / f"{utt}.wav"
        if not baseline_wav.exists() or not hybrid_wav.exists():
            continue

        smos_rows.append(
            {
                "sample_id": f"{utt}_{args.baseline_name}",
                "utterance_id": utt,
                "model": args.baseline_name,
                "prompt_wav": str(prompt_wav),
                "generated_wav": str(baseline_wav),
                "prompt_text": prompt_text,
                "target_text": gt_text,
                "smos_score": "",
                "rater_id": "",
            }
        )
        smos_rows.append(
            {
                "sample_id": f"{utt}_{args.hybrid_name}",
                "utterance_id": utt,
                "model": args.hybrid_name,
                "prompt_wav": str(prompt_wav),
                "generated_wav": str(hybrid_wav),
                "prompt_text": prompt_text,
                "target_text": gt_text,
                "smos_score": "",
                "rater_id": "",
            }
        )

        if rng.random() < 0.5:
            wav_a, wav_b = baseline_wav, hybrid_wav
            model_a, model_b = args.baseline_name, args.hybrid_name
        else:
            wav_a, wav_b = hybrid_wav, baseline_wav
            model_a, model_b = args.hybrid_name, args.baseline_name

        cmos_rows.append(
            {
                "sample_id": utt,
                "utterance_id": utt,
                "wav_a": str(wav_a),
                "wav_b": str(wav_b),
                "target_text": gt_text,
                "preferred_side": "",
                "cmos_score": "",
                "rater_id": "",
            }
        )
        key_rows.append(
            {
                "sample_id": utt,
                "utterance_id": utt,
                "system_a_model": model_a,
                "system_b_model": model_b,
                "wav_a": str(wav_a),
                "wav_b": str(wav_b),
            }
        )

    if not smos_rows:
        raise RuntimeError("No overlapping utterances found between baseline and hybrid directories.")

    with open(smos_manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(smos_rows[0].keys()))
        writer.writeheader()
        writer.writerows(smos_rows)

    with open(cmos_blind_manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cmos_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cmos_rows)

    with open(cmos_key, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"Prepared SMOS manifest: {smos_manifest}")
    print(f"Prepared CMOS blind manifest: {cmos_blind_manifest}")
    print(f"Prepared CMOS key: {cmos_key}")
    print(f"Total overlapping utterances: {len(key_rows)}")


if __name__ == "__main__":
    main()
