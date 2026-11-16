"""Original Conv2DV2 mapping deployed through this provider's own vendor."""
from .common import candidate, new_binding


def factory(case):
    import torch

    binding = new_binding()
    params = case.params

    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)

    stride, padding, dilation = [pair(params[k]) for k in ("stride", "padding", "dilation")]

    def run(x, weight, bias=None):
        x, weight = x.contiguous(), weight.contiguous()
        shape = (x.shape[0], weight.shape[0], *[
            (x.shape[i + 2] + 2 * padding[i] - dilation[i] * (weight.shape[i + 2] - 1) - 1)
            // stride[i] + 1 for i in range(2)])
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        binding.call("Conv2DV2", [x, weight, bias, None], [out], {
            "strides": [1, 1, *stride],
            "pads": [padding[0], padding[0], padding[1], padding[1]],
            "dilations": [1, 1, *dilation], "groups": int(params["groups"]),
            "data_format": "NCHW", "pad_mode": "SPECIFIC", "enable_hf32": False,
        }, formats=[0, 0, 2, 2, 0])
        return out

    return candidate(binding, run, "Conv2DV2",
                      "NCHW tensors and all original manifest convolution attributes.",
                      "conv/conv2d_v2")
