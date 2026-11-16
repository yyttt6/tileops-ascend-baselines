# aclnn Return Codes

[Table 1](#table1) describes the common return codes. To obtain error messages, call the `aclGetRecentErrMsg` API described in *Runtime APIs*. You can rectify faults based on the error messages or contact technical support.

**Table 1** Return status codes <a name='table1'></a>

| Status Code| Value| Description|
| --- | --- | --- |
| ACLNN_SUCCESS | 0 | Success.|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter verification error. Parameters contain invalid `nullptr`.|
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter verification error. For example, the two input data types do not conform to the type inference rules.|
| ACLNN_ERR_RUNTIME_ERROR | 361001 | Internal API call to NPU runtime failed.|
| ACLNN_ERR_INNER_XXX | 561xxx | API internal error. The possible causes are listed in [Table 2](#table2).|

**Table 2** Error codes <a name='table2'></a>

| Error Code| Value| Description|
| --- | --- | --- |
| ACLNN_ERR_INNER | 561000 | Internal error. An internal error occurs in an API.|
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 561001 | Internal error. An error occurs during output shape inference in an API.|
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Internal error. An error occurs when an API performs NPU kernel tiling.|
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR | 561003 | Internal error. An error occurs when an API searches for the NPU kernel (possibly because the operator binary package is not installed).|
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | Internal error. Failed to create `aclOpExecutor` in an API (possibly due to an OS exception).|
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR | 561102 | Internal error. An API does not call `uniqueExecutor ReleaseTo`.|
| ACLNN_ERR_INNER_NULLPTR | 561103 | Internal error. An error occurs in an `aclnn` API due to a `nullptr` access.|
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE | 561104 | Internal error. In an `aclnn` API, the number of operator attributes is incorrect.|
| ACLNN_ERR_INNER_KEY_CONFILICT | 561105 | This error code has been **deprecated**. Use the latest `ACLNN_ERR_INNER_KEY_CONFLICT` instead.|
| ACLNN_ERR_INNER_KEY_CONFLICT | 561105 | Internal error. In an `aclnn` API, a conflict occurs in the hash key used for operator kernel matching.|
| ACLNN_ERR_INNER_INVALID_IMPL_MODE | 561106 | Internal error. In an `aclnn` API, the operator's implementation mode parameter is invalid.|
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | 561107 | Internal error. In an `aclnn` API, the environment variable `ASCEND_OPP_PATH` is not found.|
| ACLNN_ERR_INNER_LOAD_JSON_FAILED | 561108 | Internal error. In an `aclnn` API, the operator information JSON file failed to be loaded from the kernel library.|
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND | 561109 | Internal error. In an `aclnn` API, a field in the operator information JSON file failed to be loaded from the kernel library.|
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID | 561110 | Internal error. In an `aclnn` API, the operator information JSON file in the kernel library contains an invalid value in the `format` field.|
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID | 561111 | Internal error. In an `aclnn` API, the operator information JSON file in the kernel library contains an invalid value in the `dtype` field.|
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | 561112 | Internal error. In an `aclnn` API, the binary kernel library of an operator is not loaded.|
| ACLNN_ERR_INNER_OP_FILE_INVALID | 561113 | Internal error. An error occurs when an `aclnn` API loads fields of the operator information JSON file.|
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND | 561114 | Internal error. In an `aclnn` API, the number of attributes of an operator exceeds that in the operator information JSON file.|
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH | 561115 | Internal error. In an `aclnn` API, the number of attributes of an operator is less than that in the operator information JSON file.|
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE | 561116 | Internal error. In an `aclnn` API, an operator has more than 32 inputs.|
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL | 561117 | Internal error. In an `aclnn` API, the operator information JSON file is incomplete.|
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID | 561118 | Internal error. An error occurs when an `aclnn` API parses the workspace information in the static binary JSON file.|
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID | 561119 | Internal error. An error occurs when an `aclnn` API parses the core usage in the static binary JSON file.|
