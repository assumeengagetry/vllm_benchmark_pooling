# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class EmbeddingBenchResult:
    """One result row for a single backend and sequence length."""

    model: str
    backend: str
    seq_len_target: int
    actual_mean_seq_len: float
    num_prompts: int
    enforce_eager: Optional[bool]
    convert: Optional[str]
    total_wall_time_s: float
    total_wall_time_std_s: float
    requests_per_sec: float
    input_tokens_per_sec: float
    batch_avg_latency_s: float
    timed_iters: int
    embedding_dim: Optional[int] = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    metadata: dict[str, object]
    results: list[EmbeddingBenchResult]


@dataclass
class PromptSet:
    seq_len: int
    prompts: list[str]
    generated_mean_seq_len: float


def build_prompts(
    tokenizer: AutoTokenizer,
    num_prompts: int,
    target_token_len: int,
    seed: int,
) -> list[str]:
    """Build deterministic synthetic prompts near the model-visible length."""
    rng = random.Random(seed)
    special_token_count = get_min_model_visible_tokens(tokenizer)
    content_token_budget = max(0, target_token_len - special_token_count)
    vocab = (
        "the quick brown fox jumps over a lazy dog and then runs away fast "
        "machine learning embedding model inference serving latency throughput "
        "benchmark neural network transformer attention head layer norm weight "
        "gradient descent optimizer batch size epoch loss accuracy metric score"
    ).split()

    prompts: list[str] = []
    for _ in range(num_prompts):
        if content_token_budget == 0:
            prompts.append("")
            continue

        n_words_estimate = max(1, int(target_token_len / 1.3))
        candidate = " ".join(rng.choices(vocab, k=n_words_estimate * 2))
        token_ids = tokenizer.encode(
            candidate,
            add_special_tokens=False,
            truncation=True,
            max_length=content_token_budget,
        )

        if not token_ids:
            raise ValueError("Tokenizer produced no token IDs for synthetic text.")

        if len(token_ids) < content_token_budget:
            repeats = (content_token_budget + len(token_ids) - 1) // len(token_ids)
            token_ids = (token_ids * repeats)[:content_token_budget]
        else:
            token_ids = token_ids[:content_token_budget]

        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompt = trim_to_model_visible_length(tokenizer, prompt, target_token_len)
        prompts.append(prompt)

    return prompts


def get_min_model_visible_tokens(tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(""))


def trim_to_model_visible_length(
    tokenizer: AutoTokenizer,
    prompt: str,
    target_token_len: int,
) -> str:
    """Trim text until tokenizer.encode() is no longer over target length."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    while token_ids and len(tokenizer.encode(prompt)) > target_token_len:
        token_ids = token_ids[:-1]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    return prompt


def init_vllm(
    model: str,
    tokenizer: Optional[str],
    enforce_eager: bool,
    convert: str,
    max_model_len: Optional[int],
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    dtype: str,
    seed: int,
    trust_remote_code: bool,
) -> Any:
    """Create one vLLM pooling engine for a benchmark sweep."""
    from vllm import LLM

    llm_kwargs: dict[str, object] = {
        "model": model,
        "runner": "pooling",
        "convert": convert,
        "tokenizer": tokenizer,
        "enforce_eager": enforce_eager,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "seed": seed,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

    return LLM(**llm_kwargs)


def shutdown_vllm(llm: Any) -> None:
    """Release vLLM engine resources before the next benchmark engine starts."""
    engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if engine_core is not None:
        try:
            engine_core.shutdown()
        except Exception as exc:
            print(f"WARNING: failed to shut down vLLM engine: {exc}", file=sys.stderr)


def synchronize_cuda_if_available() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_vllm_embed(
    llm: Any,
    prompts: list[str],
    warmup_prompts: int,
    num_iters: int,
) -> tuple[list[float], int, Optional[int]]:
    """Run vLLM offline embedding and return timing and output metadata."""

    for _ in range(warmup_prompts):
        llm.embed(prompts, use_tqdm=False)
        synchronize_cuda_if_available()

    wall_times: list[float] = []
    outputs = []
    for _ in range(num_iters):
        synchronize_cuda_if_available()
        start = time.perf_counter()
        outputs = llm.embed(prompts, use_tqdm=False)
        synchronize_cuda_if_available()
        wall_times.append(time.perf_counter() - start)

    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    embedding_dim: Optional[int] = None
    if outputs:
        embedding_dim = len(outputs[0].outputs.embedding)

    return (
        wall_times,
        total_input_tokens,
        embedding_dim,
    )


def build_prompt_sets(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> list[PromptSet]:
    prompt_sets: list[PromptSet] = []
    min_seq_len = get_min_model_visible_tokens(tokenizer)
    for seq_len in args.seq_lens:
        if seq_len < min_seq_len:
            raise ValueError(
                f"--seq-lens contains {seq_len}, but tokenizer requires at "
                f"least {min_seq_len} tokens after adding special tokens."
            )

        print(f"Generating prompts: seq_len={seq_len}, n={args.num_prompts}")
        prompts = build_prompts(
            tokenizer=tokenizer,
            num_prompts=args.num_prompts,
            target_token_len=seq_len,
            seed=args.seed,
        )
        token_counts = [
            len(tokenizer.encode(prompt))
            for prompt in prompts
        ]
        prompt_sets.append(
            PromptSet(
                seq_len=seq_len,
                prompts=prompts,
                generated_mean_seq_len=float(np.mean(token_counts)),
            )
        )
    return prompt_sets


def run_sentence_transformers_embed(
    st_model: Any,
    prompts: list[str],
    batch_size: int,
    warmup_prompts: int,
    num_iters: int,
) -> tuple[list[float], int, Optional[int]]:
    """Run sentence-transformers embedding as an optional baseline."""
    for _ in range(warmup_prompts):
        st_model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        synchronize_cuda_if_available()

    wall_times: list[float] = []
    embeddings = None
    for _ in range(num_iters):
        synchronize_cuda_if_available()
        start = time.perf_counter()
        embeddings = st_model.encode(
            prompts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        synchronize_cuda_if_available()
        wall_times.append(time.perf_counter() - start)

    total_input_tokens = 0
    tokenizer = getattr(st_model, "tokenizer", None)
    if tokenizer is not None:
        for prompt in prompts:
            encoded = tokenizer(prompt, truncation=True, return_tensors=None)
            total_input_tokens += len(encoded["input_ids"])

    embedding_dim = (
        embeddings.shape[1]
        if embeddings is not None and embeddings.ndim == 2
        else None
    )
    return (
        wall_times,
        total_input_tokens,
        embedding_dim,
    )


def get_sentence_transformers_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_sentence_transformers(model: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for --compare-st. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model, device=device)


def make_result(
    *,
    model: str,
    backend: str,
    seq_len: int,
    num_prompts: int,
    enforce_eager: Optional[bool],
    convert: Optional[str],
    wall_times: list[float],
    total_input_tokens: int,
    embedding_dim: Optional[int],
    extra: Optional[dict[str, object]] = None,
) -> EmbeddingBenchResult:
    if not wall_times:
        raise ValueError("wall_times must contain at least one timed iteration")

    total_wall_time_s = float(np.median(wall_times))
    total_wall_time_std_s = float(np.std(wall_times))
    if total_wall_time_s <= 0:
        raise ValueError("Timed benchmark iteration must take more than 0 seconds")

    requests_per_sec = num_prompts / total_wall_time_s
    input_tokens_per_sec = total_input_tokens / total_wall_time_s
    batch_avg_latency_s = total_wall_time_s / num_prompts
    actual_mean_seq_len = total_input_tokens / num_prompts

    return EmbeddingBenchResult(
        model=model,
        backend=backend,
        seq_len_target=seq_len,
        actual_mean_seq_len=actual_mean_seq_len,
        num_prompts=num_prompts,
        enforce_eager=enforce_eager,
        convert=convert,
        total_wall_time_s=total_wall_time_s,
        total_wall_time_std_s=total_wall_time_std_s,
        requests_per_sec=requests_per_sec,
        input_tokens_per_sec=input_tokens_per_sec,
        batch_avg_latency_s=batch_avg_latency_s,
        timed_iters=len(wall_times),
        embedding_dim=embedding_dim,
        extra={
            **(extra or {}),
            "wall_times_s": wall_times,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline throughput benchmark for vLLM embedding models. "
            "This measures LLM.embed() across controlled sequence lengths."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path. Defaults to --model.",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help=(
            "Target prompt lengths in model-visible tokens, including "
            "special tokens. Default: 64 128 256 512"
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=500,
        help="Number of prompts per benchmark cell. Default: 500",
    )
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=1,
        help="Number of full-batch untimed warmup iterations. Default: 1",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=3,
        help="Number of timed iterations per benchmark cell. Default: 3",
    )
    parser.add_argument(
        "--sweep-eager",
        action="store_true",
        help="Benchmark both enforce_eager=True and enforce_eager=False.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture when --sweep-eager is not set.",
    )
    parser.add_argument(
        "--convert",
        type=str,
        default="auto",
        choices=["auto", "none", "embed"],
        help="vLLM pooling conversion mode. Default: auto",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Model weight dtype. Default: auto",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override model maximum sequence length.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences per engine iteration.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of batched tokens per engine iteration.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use for vLLM. Default: 0.9",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size. Default: 1",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer and vLLM.",
    )
    parser.add_argument(
        "--compare-st",
        action="store_true",
        help="Also benchmark sentence-transformers on the same prompts.",
    )
    parser.add_argument(
        "--st-batch-size",
        type=int,
        default=256,
        help="sentence-transformers batch size. Default: 256",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON benchmark results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation. Default: 42",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each result row as it completes.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be greater than 0")
    if args.num_warmup_iters < 0:
        raise ValueError("--num-warmup-iters must be non-negative")
    if args.num_iters <= 0:
        raise ValueError("--num-iters must be greater than 0")
    if any(seq_len <= 0 for seq_len in args.seq_lens):
        raise ValueError("--seq-lens values must be greater than 0")
    if args.max_model_len is not None and max(args.seq_lens) > args.max_model_len:
        raise ValueError("--seq-lens values must be <= --max-model-len")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be greater than 0")
    if (
        args.max_num_batched_tokens is not None
        and args.max_num_batched_tokens <= 0
    ):
        raise ValueError("--max-num-batched-tokens must be greater than 0")
    if (
        args.max_num_batched_tokens is not None
        and max(args.seq_lens) > args.max_num_batched_tokens
    ):
        raise ValueError(
            "--max-num-batched-tokens must be >= the largest --seq-lens value"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.st_batch_size <= 0:
        raise ValueError("--st-batch-size must be greater than 0")
    if args.compare_st and importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError(
            "sentence-transformers is required for --compare-st. "
            "Install it with: pip install sentence-transformers"
        )


def print_result_table(results: list[EmbeddingBenchResult]) -> None:
    col_widths = [24, 8, 9, 8, 8, 12, 12, 14, 10, 9]
    headers = [
        "backend",
        "seq_len",
        "actual",
        "eager",
        "n",
        "req/s",
        "ktok/s",
        "avg_lat_ms",
        "std_ms",
        "emb_dim",
    ]
    header = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep = "-" * len(header)

    print()
    print("Embedding Benchmark Results")
    print(sep)
    print(header)
    print(sep)
    for result in results:
        row = [
            result.backend[: col_widths[0]],
            str(result.seq_len_target),
            f"{result.actual_mean_seq_len:.1f}",
            str(result.enforce_eager),
            str(result.num_prompts),
            f"{result.requests_per_sec:.1f}",
            f"{result.input_tokens_per_sec / 1000:.1f}",
            f"{result.batch_avg_latency_s * 1000:.2f}",
            f"{result.total_wall_time_std_s * 1000:.2f}",
            str(result.embedding_dim),
        ]
        print("  ".join(value.ljust(width) for value, width in zip(row, col_widths)))
    print(sep)
    print("avg_lat_ms is total batch wall time divided by prompt count.")
    print("Rows report the median timed iteration; std_ms is across iterations.")
    print("Use an online serving benchmark for true per-request tail latency.")


def run_vllm_sweep(
    args: argparse.Namespace,
    prompt_sets: list[PromptSet],
) -> list[EmbeddingBenchResult]:
    eager_values = [True, False] if args.sweep_eager else [args.enforce_eager]
    results: list[EmbeddingBenchResult] = []

    for enforce_eager in eager_values:
        if args.verbose:
            print(
                "Initializing vLLM: "
                f"enforce_eager={enforce_eager}, convert={args.convert}"
            )
        llm = init_vllm(
            model=args.model,
            tokenizer=args.tokenizer,
            enforce_eager=enforce_eager,
            convert=args.convert,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
        )

        try:
            for prompt_set in prompt_sets:
                if args.verbose:
                    print(
                        "Running vLLM: "
                        f"seq_len={prompt_set.seq_len}, "
                        f"enforce_eager={enforce_eager}"
                    )

                wall_times, total_tokens, embedding_dim = run_vllm_embed(
                    llm=llm,
                    prompts=prompt_set.prompts,
                    warmup_prompts=args.num_warmup_iters,
                    num_iters=args.num_iters,
                )
                result = make_result(
                    model=args.model,
                    backend="vllm",
                    seq_len=prompt_set.seq_len,
                    num_prompts=len(prompt_set.prompts),
                    enforce_eager=enforce_eager,
                    convert=args.convert,
                    wall_times=wall_times,
                    total_input_tokens=total_tokens,
                    embedding_dim=embedding_dim,
                    extra={
                        "generated_mean_seq_len": prompt_set.generated_mean_seq_len,
                    },
                )
                results.append(result)

                if args.verbose:
                    print(
                        f"  vLLM: {result.requests_per_sec:.1f} req/s, "
                        f"{result.input_tokens_per_sec / 1000:.1f} ktok/s"
                    )
        finally:
            try:
                shutdown_vllm(llm)
            finally:
                del llm
                gc.collect()

    return results


def run_sentence_transformers_sweep(
    args: argparse.Namespace,
    prompt_sets: list[PromptSet],
) -> list[EmbeddingBenchResult]:
    results: list[EmbeddingBenchResult] = []
    if args.compare_st:
        device = get_sentence_transformers_device()
        st_model = init_sentence_transformers(args.model, device)
        for prompt_set in prompt_sets:
            if args.verbose:
                print(
                    "Running sentence-transformers: "
                    f"seq_len={prompt_set.seq_len}"
                )
            wall_times, total_tokens, embedding_dim = run_sentence_transformers_embed(
                st_model=st_model,
                prompts=prompt_set.prompts,
                batch_size=args.st_batch_size,
                warmup_prompts=args.num_warmup_iters,
                num_iters=args.num_iters,
            )
            result = make_result(
                model=args.model,
                backend=f"sentence-transformers(bs={args.st_batch_size})",
                seq_len=prompt_set.seq_len,
                num_prompts=len(prompt_set.prompts),
                enforce_eager=None,
                convert=None,
                wall_times=wall_times,
                total_input_tokens=total_tokens,
                embedding_dim=embedding_dim,
                extra={
                    "device": device,
                    "generated_mean_seq_len": prompt_set.generated_mean_seq_len,
                },
            )
            results.append(result)

            if args.verbose:
                print(
                    "  sentence-transformers: "
                    f"{result.requests_per_sec:.1f} req/s, "
                    f"{result.input_tokens_per_sec / 1000:.1f} ktok/s"
                )

        del st_model
        gc.collect()

    return results


def build_metadata(args: argparse.Namespace) -> dict[str, object]:
    metadata: dict[str, object] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "seq_lens": args.seq_lens,
        "num_prompts": args.num_prompts,
        "num_warmup_iters": args.num_warmup_iters,
        "num_iters": args.num_iters,
        "convert": args.convert,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "compare_st": args.compare_st,
        "st_batch_size": args.st_batch_size,
        "seed": args.seed,
    }

    try:
        import vllm
        metadata["vllm_version"] = vllm.__version__
    except ImportError:
        pass

    try:
        import transformers
        metadata["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import torch
        metadata["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            metadata["gpu_count"] = torch.cuda.device_count()
            metadata["gpu_names"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    return metadata


def main() -> None:
    args = parse_args()
    validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )

    prompt_sets = build_prompt_sets(args, tokenizer)
    all_results = run_vllm_sweep(args, prompt_sets)
    all_results.extend(run_sentence_transformers_sweep(args, prompt_sets))
    report = BenchmarkReport(
        metadata=build_metadata(args),
        results=all_results,
    )

    print_result_table(all_results)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
