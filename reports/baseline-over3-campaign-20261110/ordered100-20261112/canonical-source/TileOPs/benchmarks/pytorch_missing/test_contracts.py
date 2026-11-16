"""Independent small-domain regression tests beyond the manifest examples.

Run without repository GPU conftests:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:. python -m pytest -q \
    --confcutdir=benchmarks/pytorch_missing benchmarks/pytorch_missing/test_contracts.py
"""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.pytorch_missing.attention import (
    dsa_paged,
    mla_cache_write,
    mla_paged,
    paged_cache_gather,
)
from benchmarks.pytorch_missing.linear import deltanet_chunk_bwd, gla_chunk_bwd, selective_scan
from benchmarks.pytorch_missing.pointwise import fused_add_rms_quant, fused_qk_norm_rope, hadamard
from benchmarks.pytorch_missing.w4a8 import gemm_w4a8, repack_w4a8
from workloads.ascend_extended import FusedQKNormRoPECall, SelectiveScanCall
from workloads.ascend_extended_attention import MLAKVCacheWriteCall, MLAPagedCall
from workloads.ascend_extended_dsa import DSAPagedCall, _make_cache
from workloads.ascend_extended_transform_quant import HadamardTransformCall
from workloads.linear_attention.chunk_cpu import deltanet_saved_cpu, gla_chunk_states_cpu
from workloads.linear_attention.deltanet import deltanet_autograd_bwd_torch
from workloads.linear_attention.gla import gla_autograd_bwd_torch


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(4)


def _work(cls, **params):
    return cls(SimpleNamespace(params=params), device="cpu")


@pytest.mark.parametrize("chunk", [1, 2, 8, 16])
def test_delta_against_autograd(chunk):
    q, k = [torch.randn(1, 2, 2 * chunk, 5) * 0.15 for _ in range(2)]
    v, do = [torch.randn(1, 2, 2 * chunk, 3) * 0.15 for _ in range(2)]
    beta = torch.rand(1, 2, 2 * chunk)
    saved = deltanet_saved_cpu(q, k, v, beta, chunk)
    got = deltanet_chunk_bwd(do, q, k, v, beta, *saved, chunk_size=chunk)
    expected = deltanet_autograd_bwd_torch(do, q, k, v, beta, chunk)
    torch.testing.assert_close(got, expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("chunk", [1, 2, 8, 16])
def test_gla_initial_and_final_gradient(chunk):
    q, k = [torch.randn(2, 2 * chunk, 2, 5) * 0.1 for _ in range(2)]
    v, do = [torch.randn(2, 2 * chunk, 2, 3) * 0.1 for _ in range(2)]
    g = -torch.rand_like(q)
    initial, dht = [torch.randn(2, 2, 5, 3) * 0.2 for _ in range(2)]
    h = gla_chunk_states_cpu(k, v, g, chunk, initial)
    got = gla_chunk_bwd(
        q, k, v, g, h, do, dht, chunk_size=chunk, scale=0.37, has_initial_state=True
    )
    expected = gla_autograd_bwd_torch(
        do, q, k, v, g, chunk, scale=0.37, initial_state=initial, dht=dht
    )
    torch.testing.assert_close(got, expected, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 65])
@pytest.mark.parametrize("softplus", [True, False])
def test_scan_grouped_reset_and_initial(length, softplus):
    u, delta, z = [torch.randn(2, 6, length) * 0.1 for _ in range(3)]
    a = -torch.rand(6, 3)
    b, c = [torch.randn(2, 2, 3, length) * 0.1 for _ in range(2)]
    d, bias = torch.randn(6), torch.rand(6)
    seq = (torch.arange(length) // 7).repeat(2, 1).int()
    init = torch.randn(2, 6, 3)
    inputs = (u, delta, a, b, c, d, z, bias, seq, init)
    expected = _work(SelectiveScanCall, delta_softplus=softplus).ref_program(*inputs)
    got = selective_scan(*inputs, delta_softplus=softplus)
    torch.testing.assert_close(got, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mla_skipped_slots_and_interleaved_rope(dtype):
    latent, pe = torch.randn(5, 4).bfloat16(), torch.randn(5, 4).bfloat16()
    cache = torch.randn(3, 2, 8).to(dtype)
    slots = torch.tensor([4, -1, 0, -1, 3])
    scale, positions, trig = torch.tensor([0.75]), torch.tensor([4, 2, 1, 0, 3]), torch.randn(7, 4)
    inputs = latent, pe, cache, slots, scale, positions, trig
    w = _work(MLAKVCacheWriteCall, fuse_rope=True, rope_layout="interleaved")
    expected = w.ref_program(*inputs)
    actual = (latent, pe, cache.clone(), slots, scale, positions, trig)
    got = mla_cache_write(*actual, fuse_rope=True, rope_layout="interleaved")
    assert got.data_ptr() == actual[2].data_ptr()
    w.verification(*inputs).validator(got, expected)


def test_gather_nonmonotonic_pages_empty_sequence_and_start():
    cache = torch.arange(6 * 2 * 3).reshape(6, 2, 3).bfloat16()
    table = torch.tensor([[4, 1, 3], [2, 0, 5], [5, 2, 1]], dtype=torch.int32)
    offsets, starts = torch.tensor([0, 3, 3, 7]), torch.tensor([1, 0, 2])
    expected = torch.stack(
        (cache[4, 1], cache[1, 0], cache[1, 1], cache[2, 0], cache[2, 1], cache[1, 0], cache[1, 1])
    )
    dst = torch.empty(7, 3, dtype=torch.bfloat16)
    assert paged_cache_gather(dst, cache, table, offsets, starts) is dst
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)


@pytest.mark.parametrize("causal", [True, False])
def test_mla_variable_lengths_and_base2_lse(causal):
    q = torch.randn(2, 2, 3, 8).bfloat16()
    cache = torch.randn(5, 2, 8).to(torch.float8_e4m3fn)
    table = torch.tensor([[3, 0, 4], [2, 1, 3]], dtype=torch.int32)
    lengths, scale = torch.tensor([3, 5], dtype=torch.int32), torch.tensor([0.75])
    params = dict(kv_lora_rank=4, is_causal=causal, sm_scale=0.2)
    inputs = q, cache, table, lengths, scale
    torch.testing.assert_close(
        mla_paged(*inputs, **params),
        _work(MLAPagedCall, **params).ref_program(*inputs),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dsa_empty_and_duplicate_selection():
    q = torch.randn(2, 2, 3, 576).bfloat16()
    cache = _make_cache((4, 2, 656), torch.Generator().manual_seed(5))
    table = torch.tensor([[3, 0, 2], [1, 2, 0]], dtype=torch.int32)
    lengths = torch.tensor([4, 6], dtype=torch.int32)
    indices = torch.tensor([[[-1, -1, -1], [2, 0, 2]], [[5, 1, -1], [3, 3, 0]]], dtype=torch.int32)
    inputs = q, cache, table, lengths, indices
    got = dsa_paged(*inputs)
    expected = _work(DSAPagedCall, sm_scale=None).ref_program(*inputs)
    torch.testing.assert_close(got, expected, atol=0.016, rtol=0.016)
    assert torch.count_nonzero(got[0][0, 0]) == 0
    assert torch.isposinf(got[1][0, 0]).all()


@pytest.mark.parametrize("layout", ["neox", "interleaved"])
def test_qk_partial_rope_preserves_v(layout):
    qkv = torch.randn(7, 5 * 8).bfloat16()
    wq, wk = torch.randn(8).bfloat16(), torch.randn(8).bfloat16()
    trig, positions = torch.randn(11, 4), torch.tensor([3, 2, 1, 7, 8, 0, 4])
    params = dict(num_heads=3, num_kv_heads=1, eps=1e-6, rope_layout=layout)
    inputs = qkv, wq, wk, trig, positions
    w = _work(FusedQKNormRoPECall, **params)
    got = fused_qk_norm_rope(qkv.clone(), *inputs[1:], **params)
    w.verification(*inputs).validator(got, w.ref_program(*inputs))


@pytest.mark.parametrize("base", [1, 12, 20, 28, 40])
def test_hadamard_special_conventions(base):
    x = torch.randn(3, base * 8).half()
    expected = _work(HadamardTransformCall, base_order=base).ref_program(x)
    torch.testing.assert_close(hadamard(x, base_order=base), expected, atol=0.001, rtol=0.001)


@pytest.mark.parametrize("dtype", ["int8", "float8_e4m3fn"])
def test_quant_zero_rows_and_clamp(dtype):
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    x[1] = torch.arange(8)
    q, scale, residual = fused_add_rms_quant(
        x, torch.zeros_like(x), torch.ones(8).bfloat16(), quant_dtype=dtype, scale_ub=0.001
    )
    assert scale[0].item() == 0 and torch.count_nonzero(q[0].float()) == 0
    torch.testing.assert_close(scale[1], torch.tensor([0.001]), atol=0, rtol=0)
    torch.testing.assert_close(residual, x, atol=0, rtol=0)


def test_w4a8_every_nibble_sign_scale_rounding():
    logical = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 8)
    packed = repack_w4a8(logical)
    assert packed[0, 0].item() == 0x98 and packed[0, -1].item() == 0x76
    a = torch.arange(128).remainder(7).float()[None].to(torch.float8_e4m3fn)
    group = torch.tensor([[0.071], [-0.043], [0.0625], [0.1]], dtype=torch.bfloat16)
    channel, token = torch.tensor([0.0, 1.0, 0.7, 1.1]), torch.tensor([0.8])
    rounded = (logical.float() * group.float()).to(torch.float8_e4m3fn).float()
    expected = ((a.float() @ rounded.T) * token[:, None] * channel).bfloat16()
    torch.testing.assert_close(
        gemm_w4a8(a, packed, group, channel, token), expected, atol=0, rtol=0
    )
