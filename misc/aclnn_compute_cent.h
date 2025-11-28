
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_COMPUTE_CENT_H_
#define ACLNN_COMPUTE_CENT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnComputeCentGetWorkspaceSize
 * parameters :
 * query : required
 * l1Cent : required
 * ivfStart : required
 * ivfLen : required
 * blockClusterTable : required
 * maskEmpty : required
 * dL1CentOut : required
 * selectNprobeOut : required
 * indicesOut : required
 * blockPositionOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnComputeCentGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *l1Cent,
    const aclTensor *ivfStart,
    const aclTensor *ivfLen,
    const aclTensor *blockClusterTable,
    const aclTensor *maskEmpty,
    const aclTensor *dL1CentOut,
    const aclTensor *selectNprobeOut,
    const aclTensor *indicesOut,
    const aclTensor *blockPositionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnComputeCent
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnComputeCent(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
