# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pin the experimental TRL SDFT surface supported by Unsloth."""

from tests.version_compat._fetch import fetch_text, has_def


TRL_SDFT_REFS = ("v1.9.2", "main")


def test_sdft_config_surface():
    for ref in TRL_SDFT_REFS:
        src = fetch_text(
            "huggingface/trl", ref, "trl/experimental/sdft/sdft_config.py"
        )
        assert src is not None, f"{ref}: SDFT config module missing"
        assert has_def(src, "SDFTConfig", "class")
        for field in (
            "teacher_model_kind",
            "generate_from_teacher",
            "distillation_mode",
            "distillation_topk",
            "use_vllm",
            "vllm_enable_sleep_mode",
        ):
            assert field in src, f"{ref}: SDFTConfig.{field} missing"


def test_sdft_trainer_uses_shared_generation_and_completion_logits():
    for ref in TRL_SDFT_REFS:
        src = fetch_text(
            "huggingface/trl", ref, "trl/experimental/sdft/sdft_trainer.py"
        )
        assert src is not None, f"{ref}: SDFT trainer module missing"
        assert has_def(src, "SDFTTrainer", "class")
        for method in (
            "_prepare_inputs",
            "_setup_teacher_model",
            "_compute_teacher_student_logits",
            "_forward_logits",
        ):
            assert has_def(src, method, "func"), f"{ref}: SDFTTrainer.{method} missing"
        assert "VLLMGeneration(" in src
        assert 'model_inputs["logits_to_keep"] = logits_to_keep + 1' in src
        assert "unwrap_model_for_generation" in src
