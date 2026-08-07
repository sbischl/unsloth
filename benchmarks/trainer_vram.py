#!/usr/bin/env python3
"""Measure one real SFT, GRPO, or SDFT step at selected completion lengths.

The controller starts a fresh Python process for every table row. This is
important for vLLM/CUDA benchmarks: an OOM or a sleeping engine must not affect
the next measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


RESULT_PREFIX = "TRAINER_VRAM_RESULT="
TRAINERS = ("sft", "grpo", "sdft")
DEFAULT_BATCH = {"sft": 1, "grpo": 2, "sdft": 1}
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def comma_values(value: str, cast=str) -> list:
    """Parse a non-empty comma-separated CLI value."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    try:
        return [cast(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def expand_cases(args: argparse.Namespace) -> list[dict]:
    trainers = comma_values(args.trainers)
    unknown = sorted(set(trainers) - set(TRAINERS))
    if unknown:
        raise ValueError(f"unknown trainers: {', '.join(unknown)}")
    lengths = sorted(set(comma_values(args.completion_lengths, int)))
    ranks = comma_values(args.lora_ranks, int)
    batches = None if args.batch_sizes is None else comma_values(args.batch_sizes, int)
    if min(lengths + ranks + (batches or [1])) <= 0:
        raise ValueError("lengths, ranks, and batch sizes must be positive")

    cases = []
    for trainer in trainers:
        trainer_batches = batches or [DEFAULT_BATCH[trainer]]
        for completion, rank, batch in itertools.product(lengths, ranks, trainer_batches):
            if trainer == "grpo" and batch % args.num_generations:
                raise ValueError(
                    "GRPO batch size must be divisible by --num-generations "
                    f"({batch} is not divisible by {args.num_generations})"
                )
            cases.append(
                {
                    "trainer": trainer,
                    "completion_length": completion,
                    "lora_rank": rank,
                    "batch_size": batch,
                }
            )
    return cases


def _worker_command(args: argparse.Namespace, case: dict) -> list[str]:
    config = {
        **case,
        "model": str(Path(args.model).resolve()),
        "prompt_length": args.prompt_length,
        "num_generations": args.num_generations,
        "distillation_topk": args.distillation_topk,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "load_mode": args.load_mode,
        "seed": args.seed,
    }
    return [sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(config)]


def run_case(args: argparse.Namespace, case: dict) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UNSLOTH_VLLM_STANDBY": "1",
            "UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE": "1",
            "TRL_EXPERIMENTAL_SILENCE": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="unsloth-trainer-vram-worker-") as worker_dir:
        completed = subprocess.run(
            _worker_command(args, case),
            cwd=worker_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return {
        **case,
        "prompt_length": args.prompt_length,
        "total_length": args.prompt_length + case["completion_length"],
        "status": "error",
        "error": f"worker exited {completed.returncode} without a result",
        "worker_output_tail": completed.stdout[-4000:],
    }


def _format_gib(value) -> str:
    return "-" if value is None else f"{value / 2**30:.2f} GiB"


def print_table(results: list[dict]) -> None:
    headers = ("trainer", "batch", "rank", "prompt", "completion", "total", "status", "peak alloc", "peak reserved", "seconds")
    rows = [headers]
    for result in results:
        rows.append(
            (
                result["trainer"], str(result["batch_size"]), str(result["lora_rank"]),
                str(result["prompt_length"]), str(result["completion_length"]),
                str(result["total_length"]), result["status"],
                _format_gib(result.get("peak_allocated_bytes")),
                _format_gib(result.get("peak_reserved_bytes")),
                "-" if result.get("seconds") is None else f"{result['seconds']:.2f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for index, row in enumerate(rows):
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


def controller(args: argparse.Namespace) -> int:
    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"local model path does not exist: {model}")
    cases = expand_cases(args)
    results = []
    stopped = set()
    for case in cases:
        series = (case["trainer"], case["batch_size"], case["lora_rank"])
        if series in stopped:
            continue
        print(
            f"Running {case['trainer']} batch={case['batch_size']} rank={case['lora_rank']} "
            f"prompt={args.prompt_length} completion={case['completion_length']}...",
            flush=True,
        )
        result = run_case(args, case)
        results.append(result)
        if result["status"] == "oom" and not args.continue_after_oom:
            stopped.add(series)
    print()
    print_table(results)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(f"\nWrote {output}")
    return 1 if any(result["status"] == "error" for result in results) else 0


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def exact_text(tokenizer, target: int) -> str:
    """Create simple text whose normal tokenizer path has exactly target tokens."""
    if target < 2:
        raise ValueError("prompt/completion lengths below 2 are not supported")
    for unit in (" x", " a", " test", " hello"):
        count = target
        for _ in range(12):
            text = unit * max(1, count)
            actual = _token_count(tokenizer, text)
            if actual == target:
                return text
            count += target - actual
            if count < 1:
                break
    raise RuntimeError(f"could not construct text with exactly {target} tokens")


def _load_model(config: dict):
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"] = "1"
    from unsloth import FastModel

    mode = config["load_mode"]
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=config["prompt_length"] + config["completion_length"],
        load_in_4bit=mode == "4bit",
        load_in_8bit=False,
        load_in_16bit=mode == "16bit",
        load_in_fp8=False,
        fast_inference=True,
        max_lora_rank=config["lora_rank"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        unsloth_vllm_standby=True,
        text_only=True,
        use_exact_model_name=True,
        max_num_seqs=max(2, config["batch_size"]),
        enforce_eager=True,
        compilation_config=0,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"],
        lora_dropout=0.0,
        target_modules=list(TARGET_MODULES),
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )
    return model, tokenizer


def _snapshot_one_lora(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_B" in name:
            return name, parameter.detach().cpu().clone()
    raise RuntimeError("no trainable LoRA B parameter found")


def _assert_lora_changed(model, snapshot) -> None:
    name, before = snapshot
    current = dict(model.named_parameters())[name].detach().cpu()
    if __import__("torch").equal(before, current):
        raise RuntimeError("trainer step completed but the sampled LoRA tensor did not change")


def _actual_completion_length(trainer, requested: int, trainer_name: str) -> int:
    if trainer_name == "sft":
        return requested
    for log in reversed(trainer.state.log_history):
        for key in ("completions/mean_length", "completion_length"):
            if key in log:
                actual = int(round(float(log[key])))
                if actual != requested:
                    raise RuntimeError(
                        f"requested {requested} completion tokens but trainer produced {actual}"
                    )
                return actual
    raise RuntimeError("trainer did not report its generated completion length")


def _common_args(config: dict, output_dir: str) -> dict:
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["batch_size"],
        "gradient_accumulation_steps": 1,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "lr_scheduler_type": "constant",
        "optim": "adamw_torch",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "seed": config["seed"],
    }


def _build_trainer(config: dict, model, tokenizer, output_dir: str):
    from datasets import Dataset

    prompt = exact_text(tokenizer, config["prompt_length"])
    completion = exact_text(tokenizer, config["completion_length"])
    prompts = [prompt] * config["batch_size"]
    common = _common_args(config, output_dir)

    if config["trainer"] == "sft":
        from trl import SFTConfig, SFTTrainer

        dataset = Dataset.from_dict({"prompt": prompts, "completion": [completion] * len(prompts)})
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                **common,
                max_length=config["prompt_length"] + config["completion_length"] + 4,
                completion_only_loss=True,
                padding_free=False,
            ),
        )
        model.vllm_engine.sleep(1)
        model._trainer_vram_sleeping = True
        return trainer

    generation_kwargs = {"ignore_eos": True, "min_tokens": config["completion_length"]}
    if config["trainer"] == "grpo":
        from trl import GRPOConfig, GRPOTrainer

        def reward(completions, **_kwargs):
            return [float(index) for index, _ in enumerate(completions)]

        return GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward,
            train_dataset=Dataset.from_dict({"prompt": prompts}),
            args=GRPOConfig(
                **common,
                num_generations=config["num_generations"],
                max_completion_length=config["completion_length"],
                generation_kwargs=generation_kwargs,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
            ),
        )

    from trl.experimental.sdft import SDFTConfig, SDFTTrainer

    return SDFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_dict(
            {"prompt": prompts, "privileged_context": ["Known correct context."] * len(prompts)}
        ),
        args=SDFTConfig(
            **common,
            remove_unused_columns=False,
            teacher_model_kind="live",
            generate_from_teacher=False,
            distillation_mode="topk_logits",
            distillation_topk=config["distillation_topk"],
            distillation_alpha=0.0,
            distillation_add_tail=False,
            distillation_is_clip=None,
            num_generations=1,
            max_prompt_length=config["prompt_length"],
            max_completion_length=config["completion_length"],
            generation_kwargs=generation_kwargs,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=True,
        ),
    )


def worker(config: dict) -> int:
    import torch

    result = {
        **{key: config[key] for key in ("trainer", "batch_size", "lora_rank", "prompt_length", "completion_length")},
        "total_length": config["prompt_length"] + config["completion_length"],
    }
    try:
        torch.manual_seed(config["seed"])
        model, tokenizer = _load_model(config)
        with tempfile.TemporaryDirectory(prefix="unsloth-trainer-vram-") as output_dir:
            trainer = _build_trainer(config, model, tokenizer, output_dir)
            model = trainer.model
            snapshot = _snapshot_one_lora(model)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_result = trainer.train()
            torch.cuda.synchronize()
            result.update(
                status="pass",
                seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                loss=train_result.training_loss,
                actual_completion_length=_actual_completion_length(
                    trainer, config["completion_length"], config["trainer"]
                ),
            )
            if train_result.global_step != 1 or not math.isfinite(train_result.training_loss):
                raise RuntimeError("trainer did not complete one finite optimizer step")
            _assert_lora_changed(model, snapshot)
            if getattr(model, "_trainer_vram_sleeping", False):
                model.vllm_engine.wake_up()
    except torch.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] != "error" else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: --trainers sft,grpo,sdft --completion-lengths "
            "128,512,1024,2048 --lora-ranks 8,32"
        ),
    )
    result.add_argument("--model", help="existing local model directory")
    result.add_argument("--trainers", default="sft,grpo,sdft", help="comma-separated trainer names")
    result.add_argument("--prompt-length", type=int, default=512, help="fixed synthetic prompt length")
    result.add_argument("--completion-lengths", default="128,512,1024,2048,4096", help="comma-separated sweep")
    result.add_argument("--lora-ranks", default="8", help="comma-separated sweep")
    result.add_argument("--batch-sizes", default=None, help="optional comma-separated sweep; defaults are 1/2/1")
    result.add_argument("--num-generations", type=int, default=2)
    result.add_argument("--distillation-topk", type=int, default=20)
    result.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    result.add_argument("--load-mode", choices=("native", "4bit", "16bit"), default="native")
    result.add_argument("--seed", type=int, default=3407)
    result.add_argument("--continue-after-oom", action="store_true")
    result.add_argument("--output")
    result.add_argument("--_worker", help=argparse.SUPPRESS)
    return result


def main() -> int:
    args = parser().parse_args()
    if args._worker:
        return worker(json.loads(args._worker))
    if not args.model:
        raise SystemExit("--model is required")
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
