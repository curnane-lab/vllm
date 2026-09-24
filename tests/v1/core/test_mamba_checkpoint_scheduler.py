# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for application-directed Mamba prefix checkpoints in the V1 scheduler.

Covers:
1. Producer prefill chunks stop exactly at the declared checkpoint position and
   register the checkpoint hash as unready.
2. Consumer requests sharing the checkpoint prefix are deferred while the
   checkpoint is unready and scheduled with a cache hit at the checkpoint once
   it is marked ready.
3. Only the declared boundary is cached for requests carrying a marker.
"""

from unittest.mock import patch

import pytest
import torch
from transformers import OPTConfig

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256 as vllm_sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

# torch_npu's allocator is not a CachingDeviceAllocator, so
# torch.accelerator.empty_cache() asserts during the global test
# cleanup on NPU hosts. These are pure-CPU tests; neuter the probe.
torch.accelerator.empty_cache = lambda: None  # type: ignore[method-assign]

pytestmark = pytest.mark.cpu_test


BLOCK_SIZE = 16
CHECKPOINT_POS = 48


import os

os.environ["VLLM_MAMBA_SAME_STEP_PAIRING"] = "1"


def _create_hybrid_mamba_scheduler(
    num_blocks: int = 1000,
    block_size: int = BLOCK_SIZE,
) -> Scheduler:
    mock_cfg = OPTConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
    )
    mock_cfg.architectures = ["OPTForCausalLM"]

    with patch("vllm.config.model.get_config", return_value=mock_cfg):
        model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            seed=42,
            skip_tokenizer_init=True,
        )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=block_size,
        log_stats=True,
    )


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    mamba_checkpoint_position: int | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=5),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, vllm_sha256),
        mamba_checkpoint_position=mamba_checkpoint_position,
    )


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(vllm_sha256)


def test_scheduler_producer_checkpoint_stops_at_boundary():
    """Producer prefill chunk stops exactly at the mamba_checkpoint_position."""
    scheduler = _create_hybrid_mamba_scheduler()
    req_producer = _make_request(
        "producer_0", [10] * 100, mamba_checkpoint_position=CHECKPOINT_POS
    )

    scheduler.add_request(req_producer)
    sched_out = scheduler.schedule()

    # Producer chunk stops at checkpoint boundary (48 tokens).
    assert [r.req_id for r in sched_out.scheduled_new_reqs] == ["producer_0"]
    assert sched_out.num_scheduled_tokens["producer_0"] == CHECKPOINT_POS
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)


def test_scheduler_cross_step_checkpoint_pending_and_wakeup():
    """Consumer is deferred while the checkpoint is in flight, and resumes as
    soon as the committing forward has been dispatched (next schedule pass)."""
    scheduler = _create_hybrid_mamba_scheduler()
    tokens_consumer = [10] * CHECKPOINT_POS + [20] * 50

    req_producer = _make_request(
        "producer_0", [10] * 100, mamba_checkpoint_position=CHECKPOINT_POS
    )
    req_consumer = _make_request(
        "consumer_0",
        tokens_consumer,
        mamba_checkpoint_position=CHECKPOINT_POS,
    )

    # Step 1: same-step pairing — the producer's checkpoint chunk and the
    # consumer's suffix are scheduled in the SAME pass; the consumer inherits
    # the producer's blocks up to the checkpoint position.
    scheduler.add_request(req_producer)
    scheduler.add_request(req_consumer)
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens["producer_0"] == CHECKPOINT_POS
    assert (
        out1.num_scheduled_tokens["consumer_0"] == len(tokens_consumer) - CHECKPOINT_POS
    )
    assert req_consumer.mamba_prefix_producer_id == "producer_0"
    assert req_consumer.mamba_checkpoint_source_block_ids is not None
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req_consumer)

    # The consumer's entire prompt was already scheduled in step 1 (the
    # suffix ran as the consumer phase); without a simulated forward there is
    # nothing left for it in step 2, while the producer continues its own
    # suffix chunk.
    out2 = scheduler.schedule()
    assert (
        out2.num_scheduled_tokens.get("producer_0") == len([10] * 100) - CHECKPOINT_POS
    )
    assert "consumer_0" not in out2.num_scheduled_tokens
    assert req_consumer.num_computed_tokens == len(tokens_consumer)
    assert not scheduler.kv_cache_manager.has_unready_checkpoint(req_producer)


def test_mamba_checkpoint_only_caches_explicit_boundary():
    """With a marker, only the checkpoint boundary gets a mamba cache entry."""
    scheduler = _create_hybrid_mamba_scheduler()
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    assert isinstance(manager.kv_cache_spec, MambaSpec)
    mamba_group_id = 1

    req = _make_request("ckpt", list(range(64)), mamba_checkpoint_position=32)
    scheduler.add_request(req)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["ckpt"] == 32
    assert scheduler.kv_cache_manager.has_unready_checkpoint(req)

    # The block ending at the checkpoint is registered (but unready).
    assert (
        manager.block_pool.get_cached_block(req.block_hashes[1], [mamba_group_id])
        is None
    )
    scheduler.kv_cache_manager.mark_checkpoint_ready("ckpt")
    assert not scheduler.kv_cache_manager.has_unready_checkpoint(req)
    assert (
        manager.block_pool.get_cached_block(req.block_hashes[1], [mamba_group_id])
        is not None
    )
    # Later blocks are NOT cached: only the declared boundary is.
    assert len(req.block_hashes) == 4
    assert (
        manager.block_pool.get_cached_block(req.block_hashes[2], [mamba_group_id])
        is None
    )
    assert (
        manager.block_pool.get_cached_block(req.block_hashes[3], [mamba_group_id])
        is None
    )


def test_consumer_hit_at_checkpoint_after_ready():
    """A consumer sharing the checkpoint prefix hits exactly at the boundary."""
    scheduler = _create_hybrid_mamba_scheduler()
    req_producer = _make_request(
        "producer_0", [10] * 100, mamba_checkpoint_position=CHECKPOINT_POS
    )
    req_consumer = _make_request(
        "consumer_0",
        [10] * CHECKPOINT_POS + [20] * 50,
        mamba_checkpoint_position=CHECKPOINT_POS,
    )

    scheduler.add_request(req_producer)
    scheduler.schedule()
    scheduler.kv_cache_manager.mark_checkpoint_ready("producer_0")

    scheduler.add_request(req_consumer)
    out = scheduler.schedule()
    assert [r.req_id for r in out.scheduled_new_reqs] == ["consumer_0"]
    new_req = out.scheduled_new_reqs[0]
    assert new_req.num_computed_tokens == CHECKPOINT_POS
    # Mamba group (group id 1) block table: nulls before the checkpoint block.
    fa_ids, mamba_ids = new_req.block_ids
    mamba_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    null_block_id = mamba_manager.block_pool.null_block.block_id
    assert mamba_ids[: CHECKPOINT_POS // BLOCK_SIZE - 1] == [null_block_id] * (
        CHECKPOINT_POS // BLOCK_SIZE - 1
    )
    assert mamba_ids[CHECKPOINT_POS // BLOCK_SIZE - 1] not in (null_block_id,)
    assert fa_ids[0] not in (null_block_id,)


def test_checkpoint_alignment_with_coarser_mamba_block():
    """Checkpoint registration must dispatch on block alignment, not on
    block_size == hash_block_size: an aligned checkpoint registers the full
    block (a partial entry at a block boundary is invalid), an unaligned one
    registers a partial-hash entry."""
    scheduler = _create_hybrid_mamba_scheduler()
    # Coarsen the mamba manager's block view to 64 tokens while hashes stay
    # 16-token (mirrors --prefix-match-unit with a large hybrid block size).
    mamba_manager = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    mamba_manager.block_size = 64
    mamba_group_id = 1

    for cp in (64, 48):
        rid = f"ckpt_aligned_{cp}"
        # Distinct token ids per case: a shared sequence would legitimately
        # hit the previous case's leftover prefix-cache entries.
        req = _make_request(
            rid, [cp * 1000 + i for i in range(80)], mamba_checkpoint_position=cp
        )
        scheduler.add_request(req)
        out = scheduler.schedule()
        assert out.num_scheduled_tokens[rid] == cp, (cp, out.num_scheduled_tokens)
        scheduler.kv_cache_manager.mark_checkpoint_ready(rid)
        # The boundary hash is registered and visible after ready.
        hash_index = cp // 16 - 1
        assert (
            mamba_manager.block_pool.get_cached_block(
                req.block_hashes[hash_index], [mamba_group_id]
            )
            is not None
        )
        if cp == 64:
            # Block-aligned: the entry is the full block 0 hash, shared with FA.
            assert mamba_manager.num_cached_block[rid] == 1
            assert rid not in mamba_manager._partial_hit_reqs
        else:
            # Sub-block: CoW source recorded for the resuming flow.
            assert mamba_manager.num_cached_block[rid] == 0
            assert rid in mamba_manager._partial_hit_reqs
        scheduler.finish_requests(rid, RequestStatus.FINISHED_STOPPED)
        scheduler.kv_cache_manager.free(req)
