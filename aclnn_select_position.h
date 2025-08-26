
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SELECT_POSITION_H_
#define ACLNN_SELECT_POSITION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSelectPositionGetWorkspaceSize
 * parameters :
 * keyIds : required
 * indices : required
 * tokenPositionOut : required
 * tokenPositionLengthOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSelectPositionGetWorkspaceSize(
    const aclTensor *keyIds,
    const aclTensor *indices,
    const aclTensor *tokenPositionOut,
    const aclTensor *tokenPositionLengthOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSelectPosition
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSelectPosition(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
