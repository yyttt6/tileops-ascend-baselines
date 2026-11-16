
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_FOREACH_SUB_LIST_H_
#define ACLNN_FOREACH_SUB_LIST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnForeachSubListGetWorkspaceSize
 * parameters :
 * x1 : dynamic
 * x2 : dynamic
 * alpha : required
 * out : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubListGetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensor *alpha,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnForeachSubList
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubList(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
