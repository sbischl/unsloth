# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Runtime contracts for TRL SDFT and Unsloth generation patching."""

from __future__ import annotations

import inspect

import pytest
import torch
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


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("add_tail", [False, True])
def test_sdft_chunked_topk_matches_trl_loss_and_gradients(alpha, add_tail):
    import unsloth  # noqa: F401
    from trl.experimental.sdft.sdft_trainer import compute_topk_self_distillation_loss
    from unsloth.models.rl import _sdft_topk_chunk_loss

    generator = torch.Generator().manual_seed(3407)
    student_hidden = torch.randn(2, 7, 11, generator=generator, requires_grad=True)
    teacher_hidden = torch.randn(2, 7, 11, generator=generator)
    student_weight = torch.randn(37, 11, generator=generator, requires_grad=True)
    teacher_weight = torch.randn(37, 11, generator=generator)
    completion_ids = torch.randint(37, (2, 7), generator=generator)

    actual, actual_token_logps = _sdft_topk_chunk_loss(
        student_hidden,
        teacher_hidden,
        student_weight,
        teacher_weight,
        None,
        None,
        completion_ids,
        1.7,
        5,
        alpha,
        add_tail,
    )
    student_logits = (student_hidden @ student_weight.T) / 1.7
    teacher_logits = (teacher_hidden @ teacher_weight.T) / 1.7
    expected = compute_topk_self_distillation_loss(
        student_logits,
        teacher_logits,
        distillation_topk=5,
        distillation_alpha=alpha,
        distillation_add_tail=add_tail,
    )
    expected_token_logps = torch.gather(
        torch.log_softmax(student_logits, dim=-1), -1, completion_ids.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_token_logps, expected_token_logps)

    actual.sum().backward(retain_graph=True)
    actual_hidden_grad = student_hidden.grad.clone()
    actual_weight_grad = student_weight.grad.clone()
    student_hidden.grad = None
    student_weight.grad = None
    expected.sum().backward()
    torch.testing.assert_close(student_hidden.grad, actual_hidden_grad)
    torch.testing.assert_close(student_weight.grad, actual_weight_grad)


def test_sdft_compute_loss_gets_unsloth_topk_wrapper():
    import unsloth  # noqa: F401
    from trl.experimental.sdft import SDFTTrainer

    assert getattr(SDFTTrainer.compute_loss, "_unsloth_sdft_topk", False)


def test_sdft_ministral3_uses_decoder_without_materializing_logits():
    import unsloth  # noqa: F401
    from types import SimpleNamespace
    from unsloth.models.rl import _sdft_completion_hidden

    expected = torch.randn(1, 9, 6)

    class Ministral:
        config = SimpleNamespace(text_config=SimpleNamespace(model_type="ministral3"))
        output_head = torch.nn.Linear(6, 37, bias=False)

        def get_output_embeddings(self):
            return self.output_head

        def set_output_embeddings(self, head):
            self.output_head = head

        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            assert isinstance(self.output_head, torch.nn.Identity)
            return SimpleNamespace(logits=expected[:, -(kwargs["logits_to_keep"]):])

    model = Ministral()
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=lambda candidate: candidate),
        model_kwarg_keys={"logits_to_keep"},
    )
    actual = _sdft_completion_hidden(
        trainer,
        model,
        torch.ones(1, 9, dtype=torch.long),
        torch.ones(1, 9, dtype=torch.long),
        4,
    )
    torch.testing.assert_close(actual, expected[:, 4:8])
    assert model.output_head is Ministral.output_head


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
