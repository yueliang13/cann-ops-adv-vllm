# For Debug 
# ulimit -c unlimited
# sysctl -w kernel.core_pattern=core-%e.%p.%h.%t
# echo 'core.%t.%e.%p' | tee /proc/sys/kernel/core_pattern
# Python ***
# gdb python3 core.*

# Test
# python fused_ifa_v5_case.py --mode perf --iters 10 --warmup 5 2>&1 | tee test.log
# python fused_ifa_v5_case.py --mode acc 2>&1 | tee test.log

# System profiling
# source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 
msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization python fused_ifa_v5_case.py --mode perf --iters 10 --warmup 5 2>&1 | tee test.log

