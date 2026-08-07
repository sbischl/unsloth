# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Runtime contracts for TRL SDFT and Unsloth generation patching."""

from __future__ import annotations

import inspect

import pytest
from packaging.version import Version


trl = pytest.importorskip("trl")
if Version(trl.__version__) < Version("1.9.2"):
    pytest.skip("TRL SDFT compatibility starts at 1.9.2", allow_module_level=True)


def test_sdft_import_gets_unsloth_generation_wrapper():
    import unsloth  # noqa: F401
    from trl.experimental.sdft import sdft_trainer
    from trl.models import unwrap_model_for_generation

    assert getattr(
        unwrap_model_for_generation, "_unsloth_generation_wrapper", False
    )
    assert sdft_trainer.unwrap_model_for_generation is unwrap_model_for_generation


def test_sdft_vllm_generation_reuses_engine_and_live_lora():
    import unsloth  # noqa: F401
    from trl.generation.vllm_generation import VLLMGeneration

    init_src = inspect.getsource(VLLMGeneration._init_vllm)
    sync_src = inspect.getsource(VLLMGeneration.sync_weights)
    generate_src = inspect.getsource(VLLMGeneration.generate)

    assert "self.llm = model.vllm_engine" in init_src
    assert "self._unsloth_load_lora = model.load_lora" in init_src
    assert "shared_weights" in sync_src and "return" in sync_src
    assert "lora_request=" in generate_src
