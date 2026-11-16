# 公共接口返回码

常见返回码如下表所示，异常信息可通过《Runtime运行时 API》中aclGetRecentErrMsg接口获取，您可以根据报错提示排查问题或联系技术支持。

**表 1**  返回状态码

| 状态码名称 | 状态码值 | 状态码说明 |
| --- | --- | --- |
| ACLNN_SUCCESS | 0 | 成功。 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 参数校验错误，参数中存在非法的nullptr。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数校验错误，如输入的两个数据类型不满足输入类型推导关系。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内存调用npu runtime的接口异常。 |
| ACLNN_ERR_INNER_XXX | 561xxx | API发生内部异常。常见的内部异常场景如下：<br>   - 561101（ACLNN_ERR_INNER_CREATE_EXECUTOR）：API内部创建aclOpExecutor失败（可能因为操作系统异常）。<br>  - 561102（ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR）：API内部未调用uniqueExecutor ReleaseTo。<br>  - 561103（ACLNN_ERR_INNER_NULLPTR）：aclnn API内部发生异常，出现了nullptr的异常。 |
