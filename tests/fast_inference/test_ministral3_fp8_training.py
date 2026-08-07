# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Opt-in Ministral-3 FP8 training and post-training inference smokes.

Run each test in a fresh process because a vLLM engine intentionally owns GPU
memory for its lifetime::

    UNSLOTH_RUN_MINISTRAL3_E2E=1 pytest -q \
      tests/fast_inference/test_ministral3_fp8_training.py::test_ministral3_sft

The model path can be overridden with ``UNSLOTH_MINISTRAL3_MODEL``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch


RUN_E2E = os.environ.get("UNSLOTH_RUN_MINISTRAL3_E2E") == "1"
MODEL_PATH = os.environ.get(
    "UNSLOTH_MINISTRAL3_MODEL",
    "/home/vllm/.cache/huggingface/hub/"
    "models--mistralai--Ministral-3-3B-Instruct-2512/"
    "snapshots/7046a0e237b436c8fb4927061ab3773772e53741",
)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
SEED = 3407
MAX_LENGTH = 256
LORA_RANK = 8

pytestmark = pytest.mark.skipif(
    not RUN_E2E or not torch.cuda.is_available() or not Path(MODEL_PATH).exists(),
    reason="opt-in test requires CUDA and the cached Ministral-3 FP8 checkpoint",
)


def _load_model():
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    # This 16 GiB test GPU needs a smaller KV reservation while the FP8 base
    # model is shared with the training process.
    os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=False,
        load_in_fp8=False,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.4,
        unsloth_vllm_standby=True,
        text_only=True,
        use_exact_model_name=True,
        enforce_eager=True,
        compilation_config=0,
    )
    assert model.config.model_type == "ministral3"
    assert getattr(model.vllm_engine, "shared_weights", False)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return FastModel, model, tokenizer


def _trainable_snapshot(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _assert_updated(model, before):
    changed = [
        name
        for name, param in model.named_parameters()
        if name in before and not torch.equal(before[name], param.detach().cpu())
    ]
    assert changed, "no trainable LoRA parameter changed"


def _assert_fast_inference(model, tokenizer, method):
    print(f"MINISTRAL3_E2E_STAGE={method}:save", flush=True)
    output_dir = Path("/tmp/unsloth-ministral3-e2e") / method
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    from unsloth.models import vLLMSamplingParams

    if getattr(model, "_unsloth_e2e_vllm_sleeping", False):
        model.vllm_engine.wake_up()
        model._unsloth_e2e_vllm_sleeping = False

    prompt = "User: Reply with one word: ready?\nAssistant:"
    request = model.load_lora(str(output_dir), load_tensors=True)
    print(f"MINISTRAL3_E2E_STAGE={method}:generate", flush=True)
    outputs = model.fast_generate(
        [prompt],
        sampling_params=vLLMSamplingParams(max_tokens=8, temperature=0.0),
        lora_request=request,
        use_tqdm=False,
    )
    token_ids = outputs[0].outputs[0].token_ids
    assert token_ids, "post-training vLLM inference returned no tokens"
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "adapter_model.safetensors").exists()
    print(f"MINISTRAL3_E2E_STAGE={method}:inference-ok", flush=True)


def _report(method, started, result):
    payload = {
        "method": method,
        "seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "global_step": result.global_step,
        "training_loss": result.training_loss,
    }
    print("MINISTRAL3_E2E=" + json.dumps(payload, sort_keys=True))


def _prompts():
    return [
        "Question: What is two plus two? Answer:",
        "Question: Name the capital of France. Answer:",
    ]


def test_ministral3_sft():
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    _fast_model, model, tokenizer = _load_model()
    dataset = Dataset.from_dict(
        {
            "prompt": _prompts(),
            "completion": [
                " Four.",
                " Paris.",
            ],
        }
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="/tmp/unsloth-ministral3-e2e/sft-trainer",
            max_length=MAX_LENGTH,
            completion_only_loss=True,
            padding_free=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=1e-2,
            optim="adamw_torch",
            logging_steps=1,
            report_to="none",
            seed=SEED,
        ),
    )
    model = trainer.model
    before = _trainable_snapshot(model)
    model.vllm_engine.sleep(1)
    model._unsloth_e2e_vllm_sleeping = True
    torch.cuda.empty_cache()
    result = trainer.train()
    print("MINISTRAL3_E2E_STAGE=sft:train-ok", flush=True)
    assert result.global_step == 2 and math.isfinite(result.training_loss)
    _assert_updated(model, before)
    _assert_fast_inference(model, tokenizer, "sft")
    _report("sft", started, result)


def _length_reward(completions, **_kwargs):
    return [float(len(item)) + i / 10 for i, item in enumerate(completions)]


def test_ministral3_grpo():
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    _fast_model, model, tokenizer = _load_model()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=_length_reward,
        train_dataset=Dataset.from_dict({"prompt": _prompts()}),
        args=GRPOConfig(
            output_dir="/tmp/unsloth-ministral3-e2e/grpo-trainer",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            num_generations=2,
            max_completion_length=8,
            max_steps=2,
            learning_rate=1e-3,
            optim="adamw_torch",
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=True,
            logging_steps=1,
            report_to="none",
            seed=SEED,
        ),
    )
    model = trainer.model
    before = _trainable_snapshot(model)
    assert trainer.vllm_generation.llm is model.vllm_engine
    result = trainer.train()
    assert result.global_step == 2 and math.isfinite(result.training_loss)
    _assert_updated(model, before)
    _assert_fast_inference(model, tokenizer, "grpo")
    _report("grpo", started, result)


def test_ministral3_sdft():
    from datasets import Dataset
    from trl.experimental.sdft import SDFTConfig, SDFTTrainer

    _fast_model, model, tokenizer = _load_model()
    dataset = Dataset.from_dict(
        {
            "prompt": _prompts(),
            "privileged_context": [
                "The correct answer is four.",
                "The correct answer is Paris.",
            ],
        }
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = SDFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SDFTConfig(
            output_dir="/tmp/unsloth-ministral3-e2e/sdft-trainer",
            remove_unused_columns=False,
            teacher_model_kind="live",
            generate_from_teacher=False,
            distillation_mode="topk_logits",
            distillation_topk=20,
            distillation_alpha=0.0,
            distillation_add_tail=False,
            distillation_is_clip=None,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=1,
            max_prompt_length=96,
            max_completion_length=8,
            max_steps=1,
            learning_rate=1e-5,
            optim="adamw_torch",
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=True,
            logging_steps=1,
            report_to="none",
            seed=SEED,
        ),
    )
    model = trainer.model
    before = _trainable_snapshot(model)
    assert trainer.teacher_model is trainer.model
    assert trainer.vllm_generation.llm is model.vllm_engine
    result = trainer.train()
    assert result.global_step == 1 and math.isfinite(result.training_loss)
    _assert_updated(model, before)
    _assert_fast_inference(model, tokenizer, "sdft")
    _report("sdft", started, result)


def test_ministral3_saved_adapter_reload():
    """Load a saved adapter in a new engine/process, matching production use."""
    adapter_path = Path(os.environ.get("UNSLOTH_E2E_ADAPTER", ""))
    if not adapter_path.is_dir():
        pytest.skip("set UNSLOTH_E2E_ADAPTER to a saved adapter directory")

    _fast_model, model, _tokenizer = _load_model()
    from unsloth.models import vLLMSamplingParams

    request = model.load_lora(str(adapter_path), load_tensors=True)
    outputs = model.fast_generate(
        ["User: Reply with one word: ready?\nAssistant:"],
        sampling_params=vLLMSamplingParams(max_tokens=8, temperature=0.0),
        lora_request=request,
        use_tqdm=False,
    )
    assert outputs[0].outputs[0].token_ids
