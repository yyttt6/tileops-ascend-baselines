"""CPU-staged hypothesis only: batch uniform chunks through unchanged mean_dim."""
def build(native, chunk, fixed):
    import torch
    def run(x, offsets=None, indices=None):
        assert offsets is None and indices is None, 'uniform exact scope only'
        if not fixed:
            return torch.stack([native.mean_dim(x[:, begin:min(begin+chunk,x.shape[1])].float().contiguous(),1,dtype=torch.float32).to(x.dtype) for begin in range(0,x.shape[1],chunk)],dim=1)
        batch,length,*features=x.shape
        assert length%chunk==0, 'tail requires separately reviewed grouping'
        packed=x.float().contiguous().reshape(batch,length//chunk,chunk,*features)
        return native.mean_dim(packed,2,dtype=torch.float32).to(x.dtype)
    return run
