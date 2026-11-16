# Common API Return Codes

The below table describes the common return codes. To obtain error messages, call the `aclGetRecentErrMsg` API described in *Runtime APIs*. You can rectify faults based on the error messages or contact technical support.

**Table 1** Returned status codes

| Error Code| Value| Description|
| --- | --- | --- |
| ACLNN_SUCCESS | 0 | Success.|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter verification error. Parameters contain invalid `nullptr`.|
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter verification error. For example, the two input data types do not meet the data type deduction requirements.|
| ACLNN_ERR_RUNTIME_ERROR | 361001 | An exception occurred when the NPU Runtime API was called.|
| ACLNN_ERR_INNER_XXX | 561xxx | Internal API exception. Common internal exception scenarios are as follows:<br>   - 561101 (ACLNN_ERR_INNER_CREATE_EXECUTOR): The API fails to create `aclOpExecutor`. (Possible cause: OS exception).<br>  - 561102 (ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR): The API does not call `uniqueExecutor ReleaseTo` internally.<br>  - 561103 (ACLNN_ERR_INNER_NULLPTR): In the aclnn API, the null pointer error occurs.|
