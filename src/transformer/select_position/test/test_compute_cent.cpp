#include <iostream>
#include <vector>
#include <cstdlib>
#include "acl/acl.h"
#include "aclnn_compute_cent.h"

int main() {
    std::cout << "=== 开始测试ComputeCent算子 ===" << std::endl;
    
    // 步骤1: 初始化ACL
    std::cout << "\n[步骤1] 初始化ACL..." << std::endl;
    aclError ret = aclInit(nullptr);
    std::cout << "  ACL初始化结果: " << ret << " (0=成功, 其他=失败)" << std::endl;
    if (ret != ACL_SUCCESS) {
        std::cout << "  ❌ ACL初始化失败" << std::endl;
        return -1;
    }
    std::cout << "  ✅ ACL初始化成功" << std::endl;
    
    // 步骤2: 设置测试参数
    std::cout << "\n[步骤2] 设置测试参数..." << std::endl;
    int64_t batch_size = 1;
    int64_t qhead_num = 2;
    int64_t kvhead_num = 2;
    int64_t seq_len = 4;
    int64_t dim = 32;
    int64_t cluster_num = 8;
    int64_t k = 4;
    
    std::cout << "  参数设置:" << std::endl;
    std::cout << "    batch_size: " << batch_size << std::endl;
    std::cout << "    qhead_num: " << qhead_num << std::endl;
    std::cout << "    kvhead_num: " << kvhead_num << std::endl;
    std::cout << "    seq_len: " << seq_len << std::endl;
    std::cout << "    dim: " << dim << std::endl;
    std::cout << "    cluster_num: " << cluster_num << std::endl;
    std::cout << "    k: " << k << std::endl;
    
    // 步骤3: 创建输入数据
    std::cout << "\n[步骤3] 创建输入数据..." << std::endl;
    std::vector<uint16_t> query_data(batch_size * qhead_num * seq_len * dim, 1);
    std::vector<uint16_t> l1_cent_data(kvhead_num * cluster_num * dim, 1);
    std::vector<uint8_t> mask_empty_data(batch_size * qhead_num * seq_len * cluster_num, 0);
    
    std::cout << "  数据大小:" << std::endl;
    std::cout << "    query_data: " << query_data.size() << " 个元素" << std::endl;
    std::cout << "    l1_cent_data: " << l1_cent_data.size() << " 个元素" << std::endl;
    std::cout << "    mask_empty_data: " << mask_empty_data.size() << " 个元素" << std::endl;
    
    // 步骤4: 创建输出数据缓冲区
    std::cout << "\n[步骤4] 创建输出数据缓冲区..." << std::endl;
    std::vector<float> d_l1_cent_data(batch_size * qhead_num * seq_len * cluster_num, 0.0f);
    std::vector<int32_t> select_nprobe_data(batch_size * qhead_num * seq_len * k, 0);
    std::vector<int32_t> indices_data(batch_size * qhead_num * seq_len * k, 0);
    
    std::cout << "  输出缓冲区大小:" << std::endl;
    std::cout << "    d_l1_cent_data: " << d_l1_cent_data.size() << " 个元素" << std::endl;
    std::cout << "    select_nprobe_data: " << select_nprobe_data.size() << " 个元素" << std::endl;
    std::cout << "    indices_data: " << indices_data.size() << " 个元素" << std::endl;
    
    // 步骤5: 创建ACL Tensor形状
    std::cout << "\n[步骤5] 创建ACL Tensor形状..." << std::endl;
    std::vector<int64_t> query_shape = {batch_size, qhead_num, seq_len, dim};
    std::vector<int64_t> l1_cent_shape = {kvhead_num, cluster_num, dim};
    std::vector<int64_t> mask_empty_shape = {batch_size, qhead_num, seq_len, cluster_num};
    std::vector<int64_t> d_l1_cent_shape = {batch_size, qhead_num, seq_len, cluster_num};
    std::vector<int64_t> select_nprobe_shape = {batch_size, qhead_num, seq_len, k};
    std::vector<int64_t> indices_shape = {batch_size, qhead_num, seq_len, k};
    
    std::cout << "  Tensor形状:" << std::endl;
    std::cout << "    query_shape: [";
    for (size_t i = 0; i < query_shape.size(); i++) {
        std::cout << query_shape[i] << (i < query_shape.size()-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    
    // 步骤6: 计算strides
    std::cout << "\n[步骤6] 计算strides..." << std::endl;
    std::vector<int64_t> query_strides(query_shape.size(), 1);
    for (int64_t i = query_shape.size() - 2; i >= 0; i--) {
        query_strides[i] = query_shape[i + 1] * query_strides[i + 1];
    }
    
    std::vector<int64_t> l1_cent_strides(l1_cent_shape.size(), 1);
    for (int64_t i = l1_cent_shape.size() - 2; i >= 0; i--) {
        l1_cent_strides[i] = l1_cent_shape[i + 1] * l1_cent_strides[i + 1];
    }
    
    std::vector<int64_t> mask_empty_strides(mask_empty_shape.size(), 1);
    for (int64_t i = mask_empty_shape.size() - 2; i >= 0; i--) {
        mask_empty_strides[i] = mask_empty_shape[i + 1] * mask_empty_strides[i + 1];
    }
    
    std::vector<int64_t> d_l1_cent_strides(d_l1_cent_shape.size(), 1);
    for (int64_t i = d_l1_cent_shape.size() - 2; i >= 0; i--) {
        d_l1_cent_strides[i] = d_l1_cent_shape[i + 1] * d_l1_cent_strides[i + 1];
    }
    
    std::vector<int64_t> select_nprobe_strides(select_nprobe_shape.size(), 1);
    for (int64_t i = select_nprobe_shape.size() - 2; i >= 0; i--) {
        select_nprobe_strides[i] = select_nprobe_shape[i + 1] * select_nprobe_strides[i + 1];
    }
    
    std::vector<int64_t> indices_strides(indices_shape.size(), 1);
    for (int64_t i = indices_shape.size() - 2; i >= 0; i--) {
        indices_strides[i] = indices_shape[i + 1] * indices_strides[i + 1];
    }
    
    std::cout << "  Strides计算完成" << std::endl;
    
    // 步骤7: 创建ACL Tensor
    std::cout << "\n[步骤7] 创建ACL Tensor..." << std::endl;
    
    std::cout << "  创建query tensor..." << std::endl;
    aclTensor* query = aclCreateTensor(query_shape.data(), query_shape.size(), ACL_FLOAT16, 
                                      query_strides.data(), 0, ACL_FORMAT_ND, 
                                      query_shape.data(), query_shape.size(), query_data.data());
    std::cout << "    query tensor创建结果: " << (query ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建l1_cent tensor..." << std::endl;
    aclTensor* l1_cent = aclCreateTensor(l1_cent_shape.data(), l1_cent_shape.size(), ACL_FLOAT16, 
                                         l1_cent_strides.data(), 0, ACL_FORMAT_ND, 
                                         l1_cent_shape.data(), l1_cent_shape.size(), l1_cent_data.data());
    std::cout << "    l1_cent tensor创建结果: " << (l1_cent ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建mask_empty tensor..." << std::endl;
    aclTensor* mask_empty = aclCreateTensor(mask_empty_shape.data(), mask_empty_shape.size(), ACL_BOOL, 
                                           mask_empty_strides.data(), 0, ACL_FORMAT_ND, 
                                           mask_empty_shape.data(), mask_empty_shape.size(), mask_empty_data.data());
    std::cout << "    mask_empty tensor创建结果: " << (mask_empty ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建d_l1_cent tensor..." << std::endl;
    aclTensor* d_l1_cent = aclCreateTensor(d_l1_cent_shape.data(), d_l1_cent_shape.size(), ACL_FLOAT, 
                                           d_l1_cent_strides.data(), 0, ACL_FORMAT_ND, 
                                           d_l1_cent_shape.data(), d_l1_cent_shape.size(), d_l1_cent_data.data());
    std::cout << "    d_l1_cent tensor创建结果: " << (d_l1_cent ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建select_nprobe tensor..." << std::endl;
    aclTensor* select_nprobe = aclCreateTensor(select_nprobe_shape.data(), select_nprobe_shape.size(), ACL_INT32, 
                                              select_nprobe_strides.data(), 0, ACL_FORMAT_ND, 
                                              select_nprobe_shape.data(), select_nprobe_shape.size(), select_nprobe_data.data());
    std::cout << "    select_nprobe tensor创建结果: " << (select_nprobe ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建indices tensor..." << std::endl;
    aclTensor* indices = aclCreateTensor(indices_shape.data(), indices_shape.size(), ACL_INT32, 
                                        indices_strides.data(), 0, ACL_FORMAT_ND, 
                                        indices_shape.data(), indices_shape.size(), indices_data.data());
    std::cout << "    indices tensor创建结果: " << (indices ? "成功" : "失败") << std::endl;
    
    // 步骤8: 创建workspace和tiling
    std::cout << "\n[步骤8] 创建workspace和tiling..." << std::endl;
    std::vector<uint8_t> workspace_data(1024, 0);
    std::vector<uint8_t> tiling_data(256, 0);
    
    std::vector<int64_t> workspace_shape = {1024};
    std::vector<int64_t> tiling_shape = {256};
    
    std::cout << "  创建workspace tensor..." << std::endl;
    aclTensor* workspace = aclCreateTensor(workspace_shape.data(), workspace_shape.size(), ACL_UINT8, 
                                           nullptr, 0, ACL_FORMAT_ND, 
                                           workspace_shape.data(), workspace_shape.size(), workspace_data.data());
    std::cout << "    workspace tensor创建结果: " << (workspace ? "成功" : "失败") << std::endl;
    
    std::cout << "  创建tiling tensor..." << std::endl;
    aclTensor* tiling = aclCreateTensor(tiling_shape.data(), tiling_shape.size(), ACL_UINT8, 
                                        nullptr, 0, ACL_FORMAT_ND, 
                                        tiling_shape.data(), tiling_shape.size(), tiling_data.data());
    std::cout << "    tiling tensor创建结果: " << (tiling ? "成功" : "失败") << std::endl;
    
    // 检查所有tensor是否创建成功
    std::cout << "\n[检查] 验证所有tensor创建状态..." << std::endl;
    if (!query || !l1_cent || !mask_empty || !d_l1_cent || !select_nprobe || !indices || !workspace || !tiling) {
        std::cout << "  ❌ 部分ACL Tensor创建失败:" << std::endl;
        std::cout << "    query: " << (query ? "✓" : "✗") << std::endl;
        std::cout << "    l1_cent: " << (l1_cent ? "✓" : "✗") << std::endl;
        std::cout << "    mask_empty: " << (mask_empty ? "✓" : "✗") << std::endl;
        std::cout << "    d_l1_cent: " << (d_l1_cent ? "✓" : "✗") << std::endl;
        std::cout << "    select_nprobe: " << (select_nprobe ? "✓" : "✗") << std::endl;
        std::cout << "    indices: " << (indices ? "✓" : "✗") << std::endl;
        std::cout << "    workspace: " << (workspace ? "✓" : "✗") << std::endl;
        std::cout << "    tiling: " << (tiling ? "✓" : "✗") << std::endl;
        return -1;
    }
    std::cout << "  ✅ 所有ACL Tensor创建成功" << std::endl;
    
    // 步骤9: 获取workspace大小
    std::cout << "\n[步骤9] 获取workspace大小..." << std::endl;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    std::cout << "  调用aclnnComputeCentGetWorkspaceSize..." << std::endl;
    std::cout << "    参数:" << std::endl;
    std::cout << "      query: " << query << std::endl;
    std::cout << "      l1_cent: " << l1_cent << std::endl;
    std::cout << "      d_l1_cent: " << d_l1_cent << std::endl;
    std::cout << "      mask_empty: " << mask_empty << std::endl;
    std::cout << "      select_nprobe: " << select_nprobe << std::endl;
    std::cout << "      indices: " << indices << std::endl;
    std::cout << "      workspace: " << workspace << std::endl;
    std::cout << "      tiling: " << tiling << std::endl;
    
    ret = aclnnComputeCentGetWorkspaceSize(
        query, l1_cent, d_l1_cent, mask_empty, select_nprobe, indices, 
        workspace, tiling, &workspaceSize, &executor);
        
    std::cout << "  aclnnComputeCentGetWorkspaceSize返回值: " << ret << std::endl;
    std::cout << "  workspaceSize: " << workspaceSize << " bytes" << std::endl;
    std::cout << "  executor: " << executor << std::endl;
    
    if (ret != ACL_SUCCESS) {
        std::cout << "  ❌ 获取workspace大小失败: " << ret << std::endl;
        return -1;
    }
    std::cout << "  ✅ 获取workspace大小成功" << std::endl;
    
    // 步骤10: 执行算子
    std::cout << "\n[步骤10] 执行算子..." << std::endl;
    aclrtStream stream = nullptr;
    std::cout << "  调用aclnnComputeCent..." << std::endl;
    std::cout << "    参数:" << std::endl;
    std::cout << "      workspace: " << workspace << std::endl;
    std::cout << "      workspaceSize: " << workspaceSize << std::endl;
    std::cout << "      executor: " << executor << std::endl;
    std::cout << "      stream: " << stream << std::endl;
    
    ret = aclnnComputeCent(workspace, workspaceSize, executor, stream);
    
    std::cout << "  aclnnComputeCent返回值: " << ret << std::endl;
    
    if (ret != ACL_SUCCESS) {
        std::cout << "  ❌ 执行算子失败: " << ret << std::endl;
        return -1;
    }
    
    std::cout << "  ✅ ComputeCent算子执行成功!" << std::endl;
    
    // 步骤11: 清理资源
    std::cout << "\n[步骤11] 清理资源..." << std::endl;
    std::cout << "  销毁query tensor..." << std::endl;
    aclDestroyTensor(query);
    std::cout << "  销毁l1_cent tensor..." << std::endl;
    aclDestroyTensor(l1_cent);
    std::cout << "  销毁mask_empty tensor..." << std::endl;
    aclDestroyTensor(mask_empty);
    std::cout << "  销毁d_l1_cent tensor..." << std::endl;
    aclDestroyTensor(d_l1_cent);
    std::cout << "  销毁select_nprobe tensor..." << std::endl;
    aclDestroyTensor(select_nprobe);
    std::cout << "  销毁indices tensor..." << std::endl;
    aclDestroyTensor(indices);
    std::cout << "  销毁workspace tensor..." << std::endl;
    aclDestroyTensor(workspace);
    std::cout << "  销毁tiling tensor..." << std::endl;
    aclDestroyTensor(tiling);
    std::cout << "  ✅ 所有tensor销毁完成" << std::endl;
    
    // 步骤12: 清理ACL
    std::cout << "\n[步骤12] 清理ACL..." << std::endl;
    aclFinalize();
    std::cout << "  ✅ ACL清理完成" << std::endl;
    
    std::cout << "\n🎉 测试完成！所有步骤都成功执行" << std::endl;
    return 0;
} 