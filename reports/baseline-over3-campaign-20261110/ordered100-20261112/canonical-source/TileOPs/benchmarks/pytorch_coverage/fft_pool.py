"""Device PyTorch candidates for FFT, adaptive max pooling and cumulative product.

No candidate copies numerical data to the host. FFT inputs use independent real
N(0, 1/2) components, the same circular Gaussian law as complex torch.randn.
"""

import math

import torch
import torch.nn.functional as F

OPS = {
    "FFTC2CFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "CumprodFwdOp",
}
MUTATED = {}


def _fft_radix2_real(x):
    """Complex FFT using real butterflies, including native float64 arithmetic.

    Ascend's torch.fft complex128 implementation attempts an unsupported double
    matmul. Bit reversal, twiddle generation and every butterfly stay on-device;
    float64 components are never converted to a narrower dtype.
    """
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("radix-two FFT requires a positive power-of-two length")
    components = torch.view_as_real(x)
    source = torch.arange(n, dtype=torch.int64, device=x.device)
    reverse = torch.zeros_like(source)
    for _ in range(n.bit_length() - 1):
        reverse = (reverse << 1) | (source & 1)
        source = source >> 1
    reordered = components.index_select(-2, reverse)
    real, imag = reordered[..., 0], reordered[..., 1]
    width = 2
    while width <= n:
        half = width // 2
        angle = torch.arange(half, dtype=real.dtype, device=x.device) * (-2 * math.pi / width)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rr = real.reshape(*x.shape[:-1], n // width, width)
        ii = imag.reshape(*x.shape[:-1], n // width, width)
        ar, ai = rr[..., :half], ii[..., :half]
        br, bi = rr[..., half:], ii[..., half:]
        tr, ti = br * cosine - bi * sine, br * sine + bi * cosine
        real = torch.stack((ar + tr, ar - tr), dim=-2).reshape(x.shape)
        imag = torch.stack((ai + ti, ai - ti), dim=-2).reshape(x.shape)
        width *= 2
    return torch.view_as_complex(torch.stack((real, imag), dim=-1))


def _adaptive_max_pool2d(x, output_size, return_indices=False):
    height, width = x.shape[-2:]
    if output_size is None:
        output_size = (None, None)
    elif isinstance(output_size, int):
        output_size = (output_size, output_size)
    oh, ow = (height if output_size[0] is None else output_size[0],
              width if output_size[1] is None else output_size[1])
    values, indices = [], []
    for row in range(oh):
        hs, he = row * height // oh, ((row + 1) * height + oh - 1) // oh
        for col in range(ow):
            ws, we = col * width // ow, ((col + 1) * width + ow - 1) // ow
            window = x[..., hs:he, ws:we].flatten(-2)
            # max selects the first finite tie. Adaptive pooling selects the
            # last NaN, whereas torch.max selects the first NaN.
            value, index = window.max(dim=-1)
            positions = torch.arange(window.shape[-1], device=x.device, dtype=torch.int64)
            last_nan = torch.where(torch.isnan(window), positions, -1).amax(dim=-1)
            index = torch.where(last_nan >= 0, last_nan, index)
            value = torch.gather(window, -1, index.unsqueeze(-1)).squeeze(-1)
            values.append(value)
            if return_indices:
                indices.append((index // (we - ws) + hs) * width + index % (we - ws) + ws)
    result = torch.stack(values, dim=-1).reshape(*x.shape[:-2], oh, ow)
    if return_indices:
        return result, torch.stack(indices, dim=-1).reshape(*x.shape[:-2], oh, ow)
    return result


def bind(case):
    op = case._call.signature.name
    if op == "FFTC2CFwdOp":
        if case._call.tensors["input"][1] == "complex128":
            return _fft_radix2_real
        return lambda x: torch.fft.fft(x, dim=-1)
    if op in ("AdaptiveMaxPool2dFwdOp", "AdaptiveMaxPool2dIndicesFwdOp"):
        size = case._call.params["output_size"]
        with_indices = op == "AdaptiveMaxPool2dIndicesFwdOp"
        return lambda x: _adaptive_max_pool2d(x, size, with_indices)
    if op == "CumprodFwdOp":
        dim = case._call.params["dim"]
        return lambda x: torch.cumprod(x.float(), dim=dim).to(x.dtype)
    raise KeyError(op)


def make_inputs(case, device):
    if case._call.signature.name != "FFTC2CFwdOp":
        return case.inputs
    shape, dtype = case._call.tensors["input"]
    real_dtype = torch.float32 if dtype == "complex64" else torch.float64
    components = torch.randn((*shape, 2), device=device, dtype=real_dtype)
    components.mul_(math.sqrt(0.5))
    return (torch.view_as_complex(components),)


def cpu_reference(case, inputs):
    op = case._call.signature.name
    x, = inputs
    if x.device.type != "cpu":
        raise ValueError("cpu_reference requires captured CPU inputs")
    if op == "FFTC2CFwdOp":
        return torch.fft.fft(x, dim=-1)
    if op in ("AdaptiveMaxPool2dFwdOp", "AdaptiveMaxPool2dIndicesFwdOp"):
        size = case._call.params["output_size"]
        return F.adaptive_max_pool2d(x, (None, None) if size is None else size,
                                     return_indices=op == "AdaptiveMaxPool2dIndicesFwdOp")
    if op == "CumprodFwdOp":
        return torch.cumprod(x.float(), dim=case._call.params["dim"]).to(x.dtype)
    raise KeyError(op)
