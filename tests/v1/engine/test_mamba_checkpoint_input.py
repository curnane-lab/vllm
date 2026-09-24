# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mamba checkpoint marker extraction in the input processor."""

from types import SimpleNamespace

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.engine.input_processor import InputProcessor

# torch_npu's allocator is not a CachingDeviceAllocator, so
# torch.accelerator.empty_cache() asserts during the global test
# cleanup on NPU hosts. These are pure-CPU tests; neuter the probe.
torch.accelerator.empty_cache = lambda: None  # type: ignore[method-assign]


class _CheckpointTokenizer:
    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if token == "<|mamba_checkpoint|>":
            return [99]
        return [1, 2]


def _checkpoint_input_processor() -> SimpleNamespace:
    return SimpleNamespace(
        mamba_checkpoint_token_id=99,
        cache_config=SimpleNamespace(
            mamba_checkpoint_token="<|mamba_checkpoint|>",
            prefix_match_unit=4,
            block_size=4,
        ),
        renderer=SimpleNamespace(tokenizer=_CheckpointTokenizer()),
    )


def test_extract_mamba_checkpoint_removes_marker():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "token",
        "prompt_token_ids": [10, 11, 12, 13, 99, 14],
    }

    updated, position = InputProcessor._extract_mamba_checkpoint(
        processor, decoder_input
    )

    assert position == 4
    assert updated["prompt_token_ids"] == [10, 11, 12, 13, 14]
    assert decoder_input["prompt_token_ids"] == [10, 11, 12, 13, 99, 14]


def test_extract_mamba_checkpoint_aligns_position_down_to_hash_unit():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "token",
        "prompt_token_ids": [10] * 7 + [99] + [14],
    }

    _, position = InputProcessor._extract_mamba_checkpoint(processor, decoder_input)
    # Marker at index 7 with hash unit 4 -> checkpoint at 4.
    assert position == 4


def test_extract_mamba_checkpoint_rejects_marker_below_hash_unit():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "multimodal",
        "prompt_token_ids": [10, 20, 21, 99, 30, 40, 41],
        "mm_kwargs": {"image": []},
        "mm_hashes": {"image": []},
        "mm_placeholders": {"image": [PlaceholderRange(offset=5, length=2)]},
    }

    # Marker at index 3 is below the hash unit (4): no valid prefix block.
    with pytest.raises(VLLMValidationError, match="too small"):
        InputProcessor._extract_mamba_checkpoint(processor, decoder_input)


def test_extract_mamba_checkpoint_shifts_placeholders_after_marker():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "multimodal",
        "prompt_token_ids": [10, 20, 21, 22, 99, 30, 40, 41],
        "mm_kwargs": {"image": []},
        "mm_hashes": {"image": []},
        "mm_placeholders": {
            "image": [
                PlaceholderRange(offset=1, length=2),
                PlaceholderRange(offset=6, length=2),
            ]
        },
    }

    updated, position = InputProcessor._extract_mamba_checkpoint(
        processor, decoder_input
    )

    assert position == 4
    assert updated["prompt_token_ids"] == [10, 20, 21, 22, 30, 40, 41]
    assert updated["mm_placeholders"]["image"] == [
        PlaceholderRange(offset=1, length=2),
        PlaceholderRange(offset=5, length=2),
    ]


def test_extract_mamba_checkpoint_rejects_marker_inside_multimodal_placeholder():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "multimodal",
        "prompt_token_ids": [10, 20, 21, 22, 99, 30],
        "mm_kwargs": {"image": []},
        "mm_hashes": {"image": []},
        "mm_placeholders": {"image": [PlaceholderRange(offset=1, length=4)]},
    }

    # The marker at index 4 sits inside the placeholder covering [1, 5).
    with pytest.raises(VLLMValidationError, match="cannot be inside"):
        InputProcessor._extract_mamba_checkpoint(processor, decoder_input)


def test_extract_mamba_checkpoint_rejects_multiple_markers():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "token",
        "prompt_token_ids": [10, 99, 11, 99, 12],
    }

    with pytest.raises(VLLMValidationError, match="exactly once"):
        InputProcessor._extract_mamba_checkpoint(processor, decoder_input)


def test_extract_mamba_checkpoint_rejects_marker_at_start_or_end():
    processor = _checkpoint_input_processor()
    for ids in ([99, 10, 11], [10, 11, 99]):
        decoder_input = {"type": "token", "prompt_token_ids": list(ids)}
        with pytest.raises(VLLMValidationError, match="both sides"):
            InputProcessor._extract_mamba_checkpoint(processor, decoder_input)


def test_extract_mamba_checkpoint_noop_without_marker():
    processor = _checkpoint_input_processor()
    decoder_input = {
        "type": "token",
        "prompt_token_ids": [10, 11, 12],
    }

    updated, position = InputProcessor._extract_mamba_checkpoint(
        processor, decoder_input
    )
    assert position is None
    assert updated["prompt_token_ids"] == [10, 11, 12]
