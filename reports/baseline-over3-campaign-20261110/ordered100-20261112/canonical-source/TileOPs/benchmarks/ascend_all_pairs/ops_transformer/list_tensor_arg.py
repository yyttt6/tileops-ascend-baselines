"""One-tensor AscendC ListTensorDesc wire layout, derived from installed SDK.

The original GroupedMatmul kernel reads this descriptor for dynamic arguments.
Only addresses and logical shapes are encoded; all operand data is unchanged.
"""

def encode_single(shape, address):
    shape = tuple(int(dim) for dim in shape)
    if not shape or any(dim < 0 for dim in shape) or not 0 <= address < 2**63:
        raise ValueError('Static positive-rank shape and valid device address required')
    # ListTensorDecode: first u64 is byte offset to the pointer array. The
    # descriptor's low 32 bits carry rank, high 32 bits carry list count=1.
    # Shape follows it, then the single original data pointer.
    return [(len(shape) + 2) * 8, (1 << 32) | len(shape), *shape, address]


class ListTensorArg:
    def __init__(self, torch, tensor):
        self.tensor = tensor
        self.words = encode_single(tensor.shape, tensor.data_ptr())
        # GE validates the supplied buffer against the logical tensor size even
        # when the original kernel interprets it as a ListTensorDesc. Allocate
        # real backing for that contract; do not advertise a larger buffer than
        # exists. The kernel reads only the descriptor words at its beginning.
        logical_bytes = tensor.numel() * tensor.element_size()
        backing_words = max(len(self.words), (logical_bytes + 7) // 8)
        self.descriptor = torch.empty((backing_words,), dtype=torch.int64,
                                      device=tensor.device)
        encoded = torch.tensor(self.words, dtype=torch.int64, device=tensor.device)
        self.descriptor[:len(self.words)].copy_(encoded)
        self.buffer_nbytes = backing_words * 8

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def data_ptr(self):
        return self.descriptor.data_ptr()
