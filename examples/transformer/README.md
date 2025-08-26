## transformer目录文件介绍

```
|—— apply_rotary_pos_emb              # ApplyRotaryPosEmb算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_apply_rotary_pos_emb.cpp # aclnnApplyRotaryPosEmb接口测试用例代码
|   |—— run_apply_rotary_pos_emb.sh   # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— dequant_rope_quant_kvcache             # DequantRopeQuantKvcache算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_dequant_rope_quant_kvcache_v2.cpp            # aclnnDequantRopeQuantKvcache接口测试用例代码
|   |—— run_dequant_rope_quant_kvcache_v2_case.sh          # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
├── ffn                        # FFN算子样例目录
|   ├── CMakeLists.txt         # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   ├── ffn_generate_data.py   # 样例数据的生成脚本，通过numpy生成随机数据，并保存到二进制文件中
|   ├── ffn_print_result.py    # 样例结果输出脚本，用于展示输出数据结果，对于float16格式的数据，避免引入复杂的c++依赖
|   ├── ffn_utils.h            # 样例代码中的公共部分，如创建aclTensor、读二进制输入数据、保存二进制输出、初始化npu资源和释放aclTensor
|   ├── test_ffn_v2.cpp        # FFNV2接口测试用例代码
|   ├── test_ffn_v3_quant.cpp  # FFNV3接口量化场景测试用例代码
|   ├── test_ffn_v3.cpp        # FFNV3接口非量化测试用例代码
|   ├── run_ffn_case.sh        # 执行CMakeLists.txt中配置的测试用例，由三个步骤组成：
|                                 1.执行ffn_generate_data.py生成输入数据二进制文件
|                                 2.执行CMakeLists.txt中配置编译的样例二进制程序
|                                 3.执行ffn_print_result.py输出测试结果
|
|—— fused_infer_attention_score       # FIA算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_fused_infer_attention_score_v2_ifa_antiquant.cpp        # FIA的IFA分支下伪量化接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_leftpad.cpp          # FIA的IFA分支下左padding接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_Lse.cpp              # FIA的IFA分支下Lse接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_PA.cpp               # FIA的IFA分支下page attention接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_postquant.cpp        # FIA的IFA分支下后量化接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_ifa_system_prefix.cpp    # FIA的IFA分支下prefix接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_innerprecise2.cpp    # FIA的PFA分支下innerprecise=2接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_left_padding.cpp     # FIA的PFA分支下左padding接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_lse.cpp              # FIA的PFA分支下lse接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_page_attention.cpp   # FIA的PFA分支下page attention接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_pse.cpp              # FIA的PFA分支下pse接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_sparse2.cpp          # FIA的PFA分支下sparse=2接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_system_prefix.cpp    # FIA的PFA分支下prefix接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_msd.cpp              # FIA的PFA分支下msd接口测试用例代码
|   |—— test_fused_infer_attention_score_v2_pfa_tensorlist.cpp       # FIA的PFA分支下tensorlist接口测试用例代码
|   |—— test_fused_infer_attention_score_v2.cpp                      # FIAV2接口测试用例代码
|   |—— test_fused_infer_attention_score.cpp                         # FIAV1接口测试用例代码
|   |—— run_fia_case.sh                                              # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
|—— incre_flash_attention             # IFA算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_incre_flash_attention.cpp               # IFAV1接口测试用例代码
|   |—— test_incre_flash_attention_v2.cpp            # IFAV2接口测试用例代码
|   |—— test_incre_flash_attention_v3.cpp            # IFAV1接口测试用例代码
|   |—— test_incre_flash_attention_v4.cpp            # IFAV1接口测试用例代码
|   |—— run_ifa_case.sh                              # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— moe_finalize_routing_v2                     # MFR算子样例目录
|   |—— CMakeLists.txt                          # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_finalize_routing_v2.cpp        # MFRV2接口测试用例代码
|   |—— run_moe_finalize_routing_v2_case.sh     # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— grouped_matmul_add                      # GMMADD算子样例目录
|   |—— CMakeLists.txt                      # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_grouped_matmul_add.cpp         # GMMADD接口测试用例代码
|   |—— run_grouped_matmul_add.sh           # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
|—— moe_gating_top_k_softmax             # MoeGatingTopKSoftmax算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_gating_top_k_softmax.cpp            # aclnnMoeGatingTopKSoftmax接口测试用例代码
|   |—— run_moe_gating_top_k_softmax_case.sh          # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)|
|
|—— moe_gating_top_k_softmax_v2             # MoeGatingTopKSoftmaxV2算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_gating_top_k_softmax_v2.cpp            # aclnnMoeGatingTopKSoftmaxV2接口测试用例代码
|   |—— run_moe_gating_top_k_softmax_v2_case.sh          # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— moe_token_unpermute_grad                       # MoeTokenUnpermuteGrad算子样例目录
|   |—— CMakeLists.txt                             # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_token_unpermute_grad.cpp          # MoeTokenUnpermuteGrad接口测试用例代码
|   |—— run_moe_token_unpermute_grad_case.sh       # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest)--disable-check-compatible(版本不对时可以加)
|
|—— moe_token_permute                 # MoeTokenPermute算子样例目录
|   |—— CMakeLists.txt                  # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_token_permute.cpp    # MoeTokenPermute接口测试用例代码
|   |—— run_moe_token_permute_case.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— moe_init_routing_quant                 # MoeInitRoutingQuantV2算子样例目录
|   |—— CMakeLists.txt                  # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_init_routing_quant.cpp    # MoeInitRoutingQuantV2接口测试用例代码
|   |—— run_moe_init_routing_quant_case.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— moe_init_routing_quant_v2                 # MoeInitRoutingQuantV2算子样例目录
|   |—— CMakeLists.txt                  # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_init_routing_quant_v2.cpp    # MoeInitRoutingQuantV2接口测试用例代码
|   |—— run_moe_init_routing_quant_v2.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— grouped_bias_add_grad                 # GroupedBiasAddGrad算子样例目录
|   |—— CMakeLists.txt                  # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_grouped_bias_add_grad.cpp    # GroupedBiasAddGrad接口测试用例代码
|   |—— run_grouped_bias_add_grad_case.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
|—— moe_compute_expert_tokens         # MoeComputeExpertTokens算子样例目录
|   |—— CMakeLists.txt                # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_moe_compute_expert_tokens.cpp    # MoeTokenPermute接口测试用例代码
|   |—— run_moe_compute_expert_tokens_case.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— moe_finalize_routing                 # MoeFinalizeRouting算子样例目录
|   |—— CMakeLists.txt                   # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— moe_finalize_routing.cpp         # MoeFinalizeRouting接口测试用例代码
|   |—— run_moe_finalize_routing_case.sh # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|—— sinkhorn                             # Sinkhorn算子样例目录
|   |—— CMakeLists.txt                   # 样例编译和添加测试用例的配置文件，由上级CMakeLists.txt调用
|   |—— test_sinkhorn.cpp                # aclnnSinkhorn接口测试用例代码
|   |—— run_sinkhorn_case.sh             # 执行CMakeLists.txt中配置的测试用例：在主目录下运行 bash build.sh build.sh -e -p xxx(装包路径到latest) --disable-check-compatible(版本不对时可以加)
|
```
