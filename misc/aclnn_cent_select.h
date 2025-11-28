
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CENT_SELECT_H_
#define ACLNN_CENT_SELECT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCentSelectGetWorkspaceSize
 * parameters :
 * query : required
 * l1Cent : required
 * blockIds : required
 * blockTable : required
 * seqLen : required
 * tokenPositionOut : required
 * tokenPositionLengthOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCentSelectGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *l1Cent,
    const aclTensor *blockIds,
    const aclTensor *blockTable,
    const aclTensor *seqLen,
    const aclTensor *pagePositionOut,
    const aclTensor *pagePositionLengthOut,
    const aclTensor *indices,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCentSelect
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCentSelect(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
