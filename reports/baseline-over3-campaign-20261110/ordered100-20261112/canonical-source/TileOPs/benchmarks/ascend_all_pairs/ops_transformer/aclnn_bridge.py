"""Original ACLNN entry points for optional and dynamic input interfaces.

This changes host API selection only. Compile/launch ownership remains required:
loading our libcust_opapi alone is not evidence of an owned numerical kernel.
"""
import ctypes as C


class HostCall:
    signatures = {
        'GroupedMatmul': ('aclnnGroupedMatmulV5', [C.c_void_p] * 12 +
                         [C.c_int64] * 4 + [C.c_void_p] * 4),
        'MhcPost': ('aclnnMhcPost', [C.c_void_p] * 5),
        'MoeFusedTopk': ('aclnnMoeFusedTopk', [C.c_void_p] * 4 +
                         [C.c_uint32] * 5 + [C.c_bool, C.c_float, C.c_bool] +
                         [C.c_void_p] * 2),
    }

    def __init__(self, op, vendor):
        from benchmarks.ascend_math_more import _MathCall
        api, signature = self.signatures[op]
        self.binding = _MathCall(api, signature, library='ops-transformer',
                                 host_library=vendor/'op_api/lib/libcust_opapi.so')
        self.op = op
        self.torch = self.binding.torch
        self.binding.codes.update({self.torch.int8: 2, self.torch.uint8: 4,
                                   self.torch.bool: 12, self.torch.float8_e5m2: 35})
        self.base = self.binding.base
        self.base.aclCreateTensorList.argtypes = [C.POINTER(C.c_void_p), C.c_uint64]
        self.base.aclCreateTensorList.restype = C.c_void_p
        self.base.aclDestroyTensorList.argtypes = [C.c_void_p]
        self.base.aclDestroyTensorList.restype = C.c_int
        self.lists = []

    def tensor(self, value, record, transpose=False):
        if value is None:
            return None
        if not value.is_contiguous():
            raise ValueError('The original storage must be packed before describing a transpose.')
        logical = value.transpose(-1, -2) if transpose else value
        shape = (C.c_int64 * logical.ndim)(*logical.shape)
        strides = (C.c_int64 * logical.ndim)(*logical.stride())
        storage = (C.c_int64 * value.ndim)(*value.shape)
        record['keep'].extend((value, logical, shape, strides, storage))
        ptr = self.base.aclCreateTensor(shape, logical.ndim,
            self.binding.codes[value.dtype], strides, 0, 2, storage, value.ndim,
            value.data_ptr())
        if not ptr:
            raise RuntimeError('aclCreateTensor returned null')
        record['tensors'].append(ptr)
        return ptr

    def tensor_list(self, value, record, transpose=False):
        if value is None:
            return None
        tensor = self.tensor(value, record, transpose)
        pointers = (C.c_void_p * 1)(tensor)
        record['keep'].append(pointers)
        ptr = self.base.aclCreateTensorList(pointers, 1)
        if not ptr:
            raise RuntimeError('aclCreateTensorList returned null')
        # ACL owns the enclosed descriptors through the list (original examples).
        record['tensors'].remove(tensor)
        self.lists.append(ptr)
        return ptr

    def call(self, inputs, outputs, attrs):
        b = self.binding
        record = b.record()
        if self.op == 'GroupedMatmul':
            if len(inputs) != 9 or len(outputs) != 1:
                raise ValueError('Expected nine original GMM input groups and one output group.')
            args = [self.tensor_list(value, record,
                    attrs.get('transpose_x', False) if i == 0 else
                    attrs.get('transpose_weight', False) if i == 1 else False)
                    for i, value in enumerate(inputs[:7])]
            args += [self.tensor_list(inputs[8], record), self.tensor(inputs[7], record),
                     None, None, None]
            args += [attrs['split_item'], attrs['group_type'], attrs['group_list_type'],
                     attrs['act_type'], b.array(attrs.get('tuning_config', (0,)), record),
                     self.tensor_list(outputs[0], record), None, None]
        elif self.op == 'MhcPost':
            args = [self.tensor(value, record) for value in [*inputs, *outputs]]
        elif self.op == 'MoeFusedTopk':
            args = [self.tensor(value, record) for value in inputs]
            args += [attrs[name] for name in ('group_num', 'group_topk', 'top_n',
                     'top_k', 'activate_type', 'is_norm', 'scale', 'enable_expert_mapping')]
            args += [self.tensor(value, record) for value in outputs]
        else:
            raise ValueError(self.op)
        b.call(args, record, outputs[0].device)

    def cleanup(self):
        self.torch.npu.synchronize()
        while self.lists:
            self.binding.check(self.base.aclDestroyTensorList(self.lists.pop()),
                               'aclDestroyTensorList')
        self.binding.cleanup()
