import torch,math
def build(fn,case):
    x,rm,rv,weight,bias=case.inputs
    assert rm is None and rv is None and case.params.get("use_input_stats",True), "requires full running-stat path"
    n,c,*spatial=x.shape
    size=math.prod(spatial)
    def run(x, rm, rv, weight, bias):
        w = torch.ones(c * size, device=x.device, dtype=x.dtype) if weight is None else weight[:, None].expand(c, size).contiguous().reshape(-1)
        b = None if bias is None else bias[:, None].expand(c, size).contiguous().reshape(-1)
        return fn(x.contiguous().reshape(n, c * size), w, b, case.params['eps'], group_size=size)[0].reshape(x.shape)
    return run
