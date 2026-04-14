import argparse
import json
import time
from contextlib import nullcontext

import torch

from f5_tts.model import CFM, DiT


BASE_ARCH = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
    attn_backend="torch",
    checkpoint_activations=False,
)

HYBRID_ARCH = dict(
    **BASE_ARCH,
    mamba_block_ids=[11, 14],
    mamba_bidirectional=True,
    mamba_d_state=64,
    mamba_d_conv=4,
    mamba_expand=1,
    mamba_alpha_init=0.02,
    mamba_output_scale_init=1e-3,
)

HYBRID_DISTILL = dict(enabled=True, hidden_weight=0.02, output_weight=0.05, hidden_layers=[11, 14, 21])


def build_model(arch: dict) -> CFM:
    return CFM(transformer=DiT(**arch, text_num_embeds=256, mel_dim=100))


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def maybe_profile_flops(model: CFM, frame_length: int, text_length: int):
    try:
        import thop
    except ImportError:
        return None

    try:
        mel = torch.randn(1, frame_length, model.num_channels)
        text = torch.zeros(1, text_length, dtype=torch.long)
        flops, _ = thop.profile(model.cpu(), inputs=(mel, text), verbose=False)
        return flops / 1e9
    except Exception:
        return None


def param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def random_batch(batch_size: int, frame_length: int, text_length: int, device: torch.device):
    mel = torch.randn(batch_size, frame_length, 100, device=device)
    text = torch.randint(0, 255, (batch_size, text_length), device=device)
    lens = torch.full((batch_size,), frame_length, device=device, dtype=torch.long)
    return mel, text, lens


def benchmark_train_step(
    model: CFM,
    *,
    device: torch.device,
    batch_size: int,
    frame_length: int,
    text_length: int,
    warmup_iters: int,
    iters: int,
    teacher_model: CFM | None = None,
):
    cleanup_cuda(device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5, weight_decay=0.01)

    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    elapsed = []
    peak_memory = None
    try:
        for step in range(warmup_iters + iters):
            mel, text, lens = random_batch(batch_size, frame_length, text_length, device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            sync(device)
            start = time.perf_counter()
            with autocast_context(device):
                if teacher_model is None:
                    loss, _, _, _ = model(mel, text=text, lens=lens, return_metadata=True)
                else:
                    loss, _, _, _ = model(
                        mel,
                        text=text,
                        lens=lens,
                        teacher_model=teacher_model,
                        distill_config=HYBRID_DISTILL,
                        return_metadata=True,
                    )
            loss.backward()
            optimizer.step()
            sync(device)
            duration = time.perf_counter() - start

            if step >= warmup_iters:
                elapsed.append(duration)
                current_peak = peak_memory_mb(device)
                if current_peak is not None:
                    peak_memory = current_peak if peak_memory is None else max(peak_memory, current_peak)
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_cuda(device)
        return {
            "train_step_latency_s": None,
            "train_step_peak_vram_mb": peak_memory,
            "train_step_oom": str(exc),
        }
    finally:
        del optimizer
        cleanup_cuda(device)

    return {
        "train_step_latency_s": sum(elapsed) / len(elapsed),
        "train_step_peak_vram_mb": peak_memory,
        "train_step_oom": None,
    }


def benchmark_inference(
    model: CFM,
    *,
    device: torch.device,
    batch_size: int,
    frame_length: int,
    text_length: int,
    sample_steps: int,
    warmup_iters: int,
    iters: int,
):
    cleanup_cuda(device)
    model = model.to(device)
    model.eval()

    cond, text, lens = random_batch(batch_size, frame_length, text_length, device)
    duration = torch.full((batch_size,), frame_length, device=device, dtype=torch.long)

    elapsed = []
    peak_memory = None
    try:
        with torch.inference_mode():
            for step in range(warmup_iters + iters):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                sync(device)
                start = time.perf_counter()
                with autocast_context(device):
                    model.sample(
                        cond=cond,
                        text=text,
                        duration=duration,
                        lens=lens,
                        steps=sample_steps,
                        cfg_strength=1.0,
                    )
                sync(device)
                duration_s = time.perf_counter() - start

                if step >= warmup_iters:
                    elapsed.append(duration_s)
                    current_peak = peak_memory_mb(device)
                    if current_peak is not None:
                        peak_memory = current_peak if peak_memory is None else max(peak_memory, current_peak)
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_cuda(device)
        return {
            "inference_latency_s": None,
            "utterances_per_second": None,
            "inference_peak_vram_mb": peak_memory,
            "inference_oom": str(exc),
        }
    finally:
        cleanup_cuda(device)

    mean_latency = sum(elapsed) / len(elapsed)
    return {
        "inference_latency_s": mean_latency,
        "utterances_per_second": batch_size / mean_latency,
        "inference_peak_vram_mb": peak_memory,
        "inference_oom": None,
    }


def benchmark_variant(
    name: str,
    model: CFM,
    *,
    device: torch.device,
    batch_size: int,
    frame_length: int,
    text_length: int,
    sample_steps: int,
    warmup_iters: int,
    iters: int,
    long_frame_length: int,
    teacher_model: CFM | None = None,
    run_train_step: bool = True,
):
    results = {
        "name": name,
        "params_m": param_count(model) / 1e6,
        "estimated_flops_g": maybe_profile_flops(model, frame_length, text_length),
    }
    if run_train_step:
        results.update(
            benchmark_train_step(
                model,
                device=device,
                batch_size=batch_size,
                frame_length=frame_length,
                text_length=text_length,
                warmup_iters=warmup_iters,
                iters=iters,
                teacher_model=teacher_model,
            )
        )
    else:
        results.update(
            {
                "train_step_latency_s": None,
                "train_step_peak_vram_mb": None,
                "train_step_oom": "skipped",
            }
        )
    results.update(
        benchmark_inference(
            model,
            device=device,
            batch_size=1,
            frame_length=frame_length,
            text_length=text_length,
            sample_steps=sample_steps,
            warmup_iters=warmup_iters,
            iters=iters,
        )
    )
    long_seq = benchmark_inference(
        model,
        device=device,
        batch_size=1,
        frame_length=long_frame_length,
        text_length=text_length,
        sample_steps=sample_steps,
        warmup_iters=warmup_iters,
        iters=iters,
    )
    results["long_sequence_inference_latency_s"] = long_seq["inference_latency_s"]
    results["long_sequence_peak_vram_mb"] = long_seq["inference_peak_vram_mb"]
    results["long_sequence_inference_oom"] = long_seq["inference_oom"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--long-frame-length", type=int, default=3072)
    parser.add_argument("--text-length", type=int, default=150)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--skip-train-step", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    baseline = build_model(BASE_ARCH)
    hybrid = build_model(HYBRID_ARCH)
    teacher = build_model(BASE_ARCH)
    teacher.requires_grad_(False)

    results = {
        "device": str(device),
        "train_step_mode": {
            "baseline": "standard_f5",
            "hybrid": "teacher_distilled_hybrid",
        },
        "baseline": benchmark_variant(
            "baseline_f5",
            baseline,
            device=device,
            batch_size=args.batch_size,
            frame_length=args.frame_length,
            text_length=args.text_length,
            sample_steps=args.sample_steps,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            long_frame_length=args.long_frame_length,
            run_train_step=not args.skip_train_step,
        ),
        "hybrid": benchmark_variant(
            "hybrid_f5_mamba",
            hybrid,
            device=device,
            batch_size=args.batch_size,
            frame_length=args.frame_length,
            text_length=args.text_length,
            sample_steps=args.sample_steps,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            long_frame_length=args.long_frame_length,
            teacher_model=teacher,
            run_train_step=not args.skip_train_step,
        ),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
