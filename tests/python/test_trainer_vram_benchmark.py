"""CPU-only tests for the trainer VRAM benchmark controller."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "benchmarks" / "trainer_vram.py"
SPEC = importlib.util.spec_from_file_location("trainer_vram", SCRIPT)
trainer_vram = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer_vram)


def _args(**overrides):
    values = {
        "trainers": "sft,grpo,sdft",
        "completion_lengths": "128,512",
        "lora_ranks": "8",
        "batch_sizes": None,
        "num_generations": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_comma_values_strips_whitespace_and_casts():
    assert trainer_vram.comma_values(" 1, 2,4 ", int) == [1, 2, 4]


def test_expand_cases_uses_trainer_specific_batch_defaults():
    cases = trainer_vram.expand_cases(_args(completion_lengths="128"))
    assert [(case["trainer"], case["batch_size"]) for case in cases] == [
        ("sft", 1),
        ("grpo", 2),
        ("sdft", 1),
    ]


def test_expand_cases_builds_requested_product():
    cases = trainer_vram.expand_cases(
        _args(trainers="sft,sdft", lora_ranks="8,32", batch_sizes="1,2")
    )
    assert len(cases) == 2 * 2 * 2 * 2


def test_grpo_rejects_incompatible_batch_size():
    with pytest.raises(ValueError, match="divisible"):
        trainer_vram.expand_cases(_args(trainers="grpo", batch_sizes="1"))


def test_print_table_contains_memory_and_status(capsys):
    trainer_vram.print_table(
        [
            {
                "trainer": "sdft",
                "batch_size": 1,
                "lora_rank": 8,
                "prompt_length": 512,
                "teacher_prompt_length": 768,
                "completion_length": 1024,
                "total_length": 1536,
                "status": "pass",
                "peak_allocated_bytes": 2**30,
                "peak_reserved_bytes": 2 * 2**30,
                "seconds": 1.25,
            }
        ]
    )
    output = capsys.readouterr().out
    assert "sdft" in output and "1.00 GiB" in output and "pass" in output


def test_actual_completion_length_checks_trainer_metric():
    trainer = type(
        "Trainer",
        (),
        {"state": type("State", (), {"log_history": [{"completions/mean_length": 128.0}]})()},
    )()
    assert trainer_vram._actual_completion_length(trainer, 128, "sdft") == 128
    with pytest.raises(RuntimeError, match="produced 128"):
        trainer_vram._actual_completion_length(trainer, 256, "sdft")
