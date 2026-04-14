"""Benchmark Phase14 volatility inference latency against CNDiff.

This script measures *inference speed only*.

- `Phase14FusionModel` is loaded from this repository and benchmarked on real
  local embedding tensors.
- `CNDiff` is loaded from `external/CNDiff` and benchmarked on synthetic input
  tensors that match its native config/checkpoint shapes.

That means the latency comparison is real, but the forecasting task is not yet
apples-to-apples. For a fair quality comparison, CNDiff must be retrained on
the same volatility target used by the local model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE14_CKPT = PROJECT_ROOT / "models" / "phase14_fusion_best.pt"
DEFAULT_PHASE14_EMB = PROJECT_ROOT / "data" / "embeddings" / "phase13_fusion_embeddings.pt"
DEFAULT_PHASE14_HAR = PROJECT_ROOT / "data" / "embeddings" / "phase14_har_rv_raw.pt"
DEFAULT_CNDIFF_ROOT = PROJECT_ROOT / "external" / "CNDiff"
DEFAULT_CNDIFF_CFG = DEFAULT_CNDIFF_ROOT / "cfg" / "exchange.yaml"
DEFAULT_CNDIFF_CKPT = DEFAULT_CNDIFF_ROOT / "saved_models" / "exchange.pth"
DEFAULT_JSON_OUT = PROJECT_ROOT / "reports" / "phase14_vs_cndiff_latency.json"


@dataclass
class BenchmarkResult:
    name: str
    device: str
    batch_size: int
    warmup_steps: int
    timed_steps: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples_per_sec: float
    param_count: int
    output_shape: list[int]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Phase14 volatility latency against CNDiff."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--phase14-offset", type=int, default=0)
    parser.add_argument("--phase14-ckpt", type=Path, default=DEFAULT_PHASE14_CKPT)
    parser.add_argument("--phase14-emb", type=Path, default=DEFAULT_PHASE14_EMB)
    parser.add_argument("--phase14-har", type=Path, default=DEFAULT_PHASE14_HAR)
    parser.add_argument("--cndiff-root", type=Path, default=DEFAULT_CNDIFF_ROOT)
    parser.add_argument("--cndiff-cfg", type=Path, default=DEFAULT_CNDIFF_CFG)
    parser.add_argument("--cndiff-ckpt", type=Path, default=DEFAULT_CNDIFF_CKPT)
    parser.add_argument(
        "--cndiff-copies",
        type=int,
        default=1,
        help="Number of stochastic CNDiff trajectories to average in the primary run.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def namespaceify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: namespaceify(val) for key, val in obj.items()})
    if isinstance(obj, list):
        return [namespaceify(item) for item in obj]
    return obj


def load_yaml_config(path: Path) -> SimpleNamespace:
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict) and "default" in raw:
        raw = raw["default"]
    return namespaceify(raw)


def load_phase14_runner(
    device: torch.device,
    checkpoint_path: Path,
    embeddings_path: Path,
    har_rv_path: Path,
    batch_size: int,
    offset: int,
) -> tuple[torch.nn.Module, callable, dict[str, Any]]:
    from src.models.fusion_model import Phase14FusionModel

    embeddings = torch.load(embeddings_path, weights_only=False, map_location="cpu")
    har_rv = torch.load(har_rv_path, weights_only=False, map_location="cpu")

    total_samples = int(embeddings["price_emb"].shape[0])
    end = offset + batch_size
    if offset < 0 or end > total_samples:
        raise ValueError(
            f"Requested Phase14 slice [{offset}:{end}] is outside 0..{total_samples}."
        )

    batch = {
        "price_emb": embeddings["price_emb"][offset:end].to(device),
        "har_rv_raw": har_rv[offset:end].to(device),
        "gat_emb": embeddings["gat_emb"][offset:end].to(device),
        "doc_emb": embeddings["doc_emb"][offset:end].to(device),
        "macro_emb": embeddings["macro_emb"][offset:end].to(device),
        "surprise_feat": embeddings["surprise_feat"][offset:end].to(device),
        "modality_mask": embeddings["modality_mask"][offset:end].to(device),
    }

    model = Phase14FusionModel(mc_dropout=False).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def runner() -> torch.Tensor:
        out = model(
            batch["price_emb"],
            batch["har_rv_raw"],
            batch["gat_emb"],
            batch["doc_emb"],
            batch["macro_emb"],
            batch["surprise_feat"],
            batch["modality_mask"],
        )
        return out["volatility_pred"]

    metadata = {
        "checkpoint": rel_path(checkpoint_path),
        "embeddings": rel_path(embeddings_path),
        "har_rv": rel_path(har_rv_path),
        "offset": offset,
        "available_samples": total_samples,
        "task": "scalar volatility regression",
    }
    return model, runner, metadata


def load_cndiff_model(
    device: torch.device,
    cndiff_root: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, SimpleNamespace]:
    root_str = str(cndiff_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from models.diffusion_model import nonlinear_conditional_ddpm

    config = load_yaml_config(config_path)
    config.train.device = "cuda:0" if device.type == "cuda" else "cpu"
    model = nonlinear_conditional_ddpm(config).to(device)
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def make_cndiff_runner(
    model: torch.nn.Module,
    config: SimpleNamespace,
    device: torch.device,
    batch_size: int,
    copies: int,
    seed: int,
) -> tuple[callable, dict[str, Any]]:
    if copies < 1:
        raise ValueError("CNDiff copies must be >= 1.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    base_x = torch.randn(
        batch_size,
        config.data.seq_len,
        config.data.feature_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    base_y = torch.zeros(
        batch_size,
        config.data.pred_len,
        config.data.feature_dim,
        dtype=torch.float32,
        device=device,
    )

    if copies == 1:
        batch_x = base_x
        batch_y = base_y
    else:
        batch_x = base_x.repeat(copies, 1, 1, 1).transpose(0, 1).flatten(0, 1).contiguous()
        batch_y = base_y.repeat(copies, 1, 1, 1).transpose(0, 1).flatten(0, 1).contiguous()

    def runner() -> torch.Tensor:
        out = model.p_sample_loop(batch_y, batch_x)
        if copies == 1:
            return out
        return out.view(
            batch_size,
            copies,
            config.data.pred_len,
            config.data.feature_dim,
        ).mean(dim=1)

    metadata = {
        "task": "native multivariate sequence generation",
        "seq_len": int(config.data.seq_len),
        "pred_len": int(config.data.pred_len),
        "feature_dim": int(config.data.feature_dim),
        "diffusion_steps": int(config.diff.timesteps),
        "copies_averaged": copies,
        "internal_batch_size": int(batch_size * copies),
    }
    return runner, metadata


def benchmark_runner(
    name: str,
    runner: callable,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iters: int,
    metadata: dict[str, Any],
) -> BenchmarkResult:
    latencies_ms: list[float] = []
    output_shape: list[int] | None = None

    with torch.inference_mode():
        for _ in range(warmup):
            _ = runner()
            sync_device(device)

        for _ in range(iters):
            sync_device(device)
            start = time.perf_counter()
            output = runner()
            sync_device(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if output_shape is None:
                output_shape = list(output.shape)

    mean_ms = statistics.fmean(latencies_ms)
    throughput = batch_size / (mean_ms / 1000.0)

    return BenchmarkResult(
        name=name,
        device=str(device),
        batch_size=batch_size,
        warmup_steps=warmup,
        timed_steps=iters,
        mean_ms=mean_ms,
        median_ms=statistics.median(latencies_ms),
        p95_ms=percentile(latencies_ms, 0.95),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        samples_per_sec=throughput,
        param_count=count_parameters(model),
        output_shape=output_shape or [],
        metadata=metadata,
    )


def print_table(results: list[BenchmarkResult]) -> None:
    print("\nInference latency benchmark")
    print("=" * 100)
    print(
        f"{'Model':<26} {'Mean ms':>10} {'P95 ms':>10} "
        f"{'Samples/s':>12} {'Params':>12} {'Output':>20}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.name:<26} "
            f"{result.mean_ms:>10.2f} "
            f"{result.p95_ms:>10.2f} "
            f"{result.samples_per_sec:>12.2f} "
            f"{result.param_count:>12,d} "
            f"{str(tuple(result.output_shape)):>20}"
        )

    if len(results) >= 2:
        reference = results[0]
        print("-" * 100)
        for result in results[1:]:
            ratio = result.mean_ms / reference.mean_ms
            print(
                f"{result.name} is {ratio:.1f}x slower than {reference.name} "
                f"at batch size {reference.batch_size}."
            )


def build_payload(
    args: argparse.Namespace,
    device: torch.device,
    results: list[BenchmarkResult],
) -> dict[str, Any]:
    comparison: dict[str, float] = {}
    if results:
        baseline = results[0]
        for result in results[1:]:
            comparison[f"{result.name}_vs_{baseline.name}_latency_ratio"] = (
                result.mean_ms / baseline.mean_ms
            )

    return {
        "device": str(device),
        "seed": args.seed,
        "fairness_note": (
            "Latency comparison is real. Task equivalence is not: Phase14 is a "
            "trained scalar volatility model, while CNDiff here is benchmarked "
            "on its native multivariate diffusion setup. Retrain CNDiff on the "
            "same volatility target before interpreting accuracy or efficiency "
            "as an apples-to-apples winner."
        ),
        "results": [asdict(result) for result in results],
        "comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(args.device)

    phase14_model, phase14_runner, phase14_meta = load_phase14_runner(
        device=device,
        checkpoint_path=args.phase14_ckpt,
        embeddings_path=args.phase14_emb,
        har_rv_path=args.phase14_har,
        batch_size=args.batch_size,
        offset=args.phase14_offset,
    )
    phase14_result = benchmark_runner(
        name="Phase14Fusion",
        runner=phase14_runner,
        model=phase14_model,
        device=device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        metadata=phase14_meta,
    )

    cndiff_model, cndiff_config = load_cndiff_model(
        device=device,
        cndiff_root=args.cndiff_root,
        config_path=args.cndiff_cfg,
        checkpoint_path=args.cndiff_ckpt,
    )

    cndiff_copy_values = [args.cndiff_copies]
    native_copies = int(getattr(cndiff_config.diff, "n_copies_to_test", 1))
    if native_copies not in cndiff_copy_values:
        cndiff_copy_values.append(native_copies)

    results = [phase14_result]
    for copies in cndiff_copy_values:
        cndiff_runner, cndiff_meta = make_cndiff_runner(
            model=cndiff_model,
            config=cndiff_config,
            device=device,
            batch_size=args.batch_size,
            copies=copies,
            seed=args.seed,
        )
        cndiff_meta["checkpoint"] = rel_path(args.cndiff_ckpt)
        cndiff_meta["config"] = rel_path(args.cndiff_cfg)
        result = benchmark_runner(
            name=f"CNDiff(copies={copies})",
            runner=cndiff_runner,
            model=cndiff_model,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
            metadata=cndiff_meta,
        )
        results.append(result)

    print_table(results)

    payload = build_payload(args=args, device=device, results=results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
