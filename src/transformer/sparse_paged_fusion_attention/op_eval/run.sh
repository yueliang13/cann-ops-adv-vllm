# msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization python ifa_torch_paged_test.py
# msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization python ifa_torch_and_cent_select.py

# 配置环境 
# source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 
# msprof op python ifa_v5_case.py
# msprof op --aic-metrics=KernelScale,Roofline,TimelineDetail,Occupancy,MemoryDetail python ifa_v5_case.py
# msprof op --aic-metrics=Source python ifa_v5_case.py

# --sys-pid-profiling=on
# msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization python ifa_v5_case.py
# export ASCEND_RT_VISIBLE_DEVICES=2
# msprof op simulator --soc-version=Ascend910B2 --output=output_data python ifa_v5_case.py # 仿真出来的结果很奇怪


source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 

eval_sparse_paged(){
    # 规范化目录命名：
    # - Batch_Eval 使用 $2
    # - Seq_Len_Eval 使用原始 $4 表达式（如 8*1024）
    # - Block_Size_Eval 使用 $3
    local SIZE_TAG="$2"
    if [ "$1" = "Seq_Len_Eval" ]; then
        SIZE_TAG="$4"
    elif [ "$1" = "Block_Size_Eval" ]; then
        SIZE_TAG="$3"
    fi

    mkdir -p "./$1/Size_${SIZE_TAG}"
    msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization \
    --output="./$1/Size_${SIZE_TAG}" \
    python fused_ifa_v5_torch_case.py \
        --mode perf \
        --iters 3 \
        --warmup 5 \
        --B $2 \
        --block_size $3 \
        --total_seq_len_kv $4 
}

eval_sparse_paged "Batch_Eval_128K" 8 128 128*1024 32 8 128 128 0

# # Batch Eval
# eval_sparse_paged "Batch_Eval" 1 128 32*1024 32 8 128 128 0
# eval_sparse_paged "Batch_Eval" 2 128 32*1024 32 8 128 128 0
# eval_sparse_paged "Batch_Eval" 4 128 32*1024 32 8 128 128 0
# eval_sparse_paged "Batch_Eval" 6 128 32*1024 32 8 128 128 0
# eval_sparse_paged "Batch_Eval" 8 128 32*1024 32 8 128 128 0

# # # Seq_Len Eval
# eval_sparse_paged "Seq_Len_Eval" 1 128 8*1024 32 8 128 128 0
# eval_sparse_paged "Seq_Len_Eval" 1 128 16*1024 32 8 128 128 0
# eval_sparse_paged "Seq_Len_Eval" 1 128 32*1024 32 8 128 128 0
# eval_sparse_paged "Seq_Len_Eval" 1 128 64*1024 32 8 128 128 0
# eval_sparse_paged "Seq_Len_Eval" 1 128 128*1024 32 8 128 128 0

# # #Block_Size Eval
# eval_sparse_paged "Block_Size_Eval" 1 16 32*1024 32 8 128 128 0
# eval_sparse_paged "Block_Size_Eval" 1 32 32*1024 32 8 128 128 0
# eval_sparse_paged "Block_Size_Eval" 1 64 32*1024 32 8 128 128 0
# eval_sparse_paged "Block_Size_Eval" 1 128 32*1024 32 8 128 128 0
