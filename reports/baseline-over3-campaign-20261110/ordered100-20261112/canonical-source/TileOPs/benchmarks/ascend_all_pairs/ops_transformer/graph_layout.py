"""Original GroupedMatmul graph metadata with explicit FP32 and packed INT4 ListTensorDesc buffers."""


class PackedInt4:
    def __init__(self, tensor):
        self.tensor = tensor
    def __getattr__(self, name):
        return getattr(self.tensor, name)
    @property
    def shape(self):
        return (*self.tensor.shape[:-1], self.tensor.shape[-1] * 8)
    @property
    def dtype(self):
        return 'packed_int4'



class TypedEmpty:
    """Describe only an absent dynamic input using its declared schema dtype."""
    def __init__(self, tensor, dtype):
        if tensor.numel() != 0:
            raise ValueError("TypedEmpty cannot reinterpret nonempty data")
        self.tensor = tensor
        self.descriptor_dtype = dtype

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    @property
    def dtype(self):
        return self.descriptor_dtype


def grouped_matmul(torch, inputs, outputs, attrs):
    """Only prepare address metadata; original kernel performs every product."""
    labels = ['x', 'weight', 'bias', 'scale', 'offset', 'antiquant_scale',
              'antiquant_offset', None, None, 'y']
    inputs = list(inputs)
    x = inputs[0]
    for index in range(2, 7):
        if inputs[index] is not None:
            continue
        # IsTensorListNullOrEmpty explicitly accepts a single [0] tensor.
        # A DT_UNDEFINED placeholder disappears from graph inference and
        # otherwise shifts ordinary group_list into the bias position.
        dtype = torch.float32
        if index == 2:
            dtype = torch.int32 if x.dtype == torch.int8 else (
                torch.float16 if x.dtype == torch.float16 else torch.float32)
        elif index in (5, 6):
            # Original quantized-weight signatures use the floating X dtype;
            # nonquantized and FP8/INT8 X signatures declare empty antiquant FP16.
            quantized_weight = inputs[1].dtype in (
                torch.int8, torch.int32, torch.float8_e4m3fn, torch.float8_e5m2)
            dtype = (x.dtype if quantized_weight and
                     x.dtype in (torch.float16, torch.bfloat16) else torch.float16)
        # Original ascend950 floating-X signatures declare absent scale UINT64.
        # No data is reinterpreted: these [0] tensors are absent to kernel tiling.
        empty = torch.empty((0,), dtype=dtype, device=x.device)
        inputs[index] = (TypedEmpty(empty, 'uint64')
                         if index == 3 and x.dtype in
                         (torch.float32, torch.float16, torch.bfloat16) else empty)
    values = [*inputs, *outputs]
    # c09 actual graph logs prove that these buffers reach the original kernel
    # without ListTensorDesc packing. Keep compile shapes/dtypes unchanged and
    # supply its exact public descriptor wire format for the FP32 route.
    packed = [PackedInt4(value) if index == 1 and value.dtype == torch.int32 else value
              for index, value in enumerate(values)]
    if ((x.dtype == torch.float32 and inputs[1].dtype == torch.float32) or
            (x.dtype in (torch.float16, torch.bfloat16) and inputs[1].dtype == torch.int32)):
        from .list_tensor_arg import ListTensorArg
        packed = [ListTensorArg(torch, value) if labels[index] is not None else value
                  for index, value in enumerate(packed)]
    attributes = dict(attrs)
    for label, value in zip(labels[:7], inputs[:7]):
        attributes['__dynamic_input_' + label + '_cnt'] = int(value is not None)
    attributes['__dynamic_output_y_cnt'] = len(outputs)
    # Each group retains its own typed descriptor, including empty groups.
    return packed[:len(inputs)], packed[len(inputs):], attributes, labels
