import torch,math
def build(fn,case):
    x,weight,bias=case.inputs
    n,c,*spatial=x.shape
    size=math.prod(spatial)
    groups=case.params["num_groups"]
    def run(x, weight, bias):
        w = torch.ones(c * size, device=x.device, dtype=x.dtype) if weight is None else weight[:, None].expand(c, size).contiguous().reshape(-1)
        b = None if bias is None else bias[:, None].expand(c, size).contiguous().reshape(-1)
        return fn(x.contiguous().reshape(n, c * size), w, b, case.params['eps'], group_size=c * size // groups)[0].reshape(x.shape)
    return run
