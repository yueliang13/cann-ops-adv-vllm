# 配置环境 
# source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 
# msprof op python ifa_v5_case.py
# msprof op --aic-metrics=KernelScale,Roofline,TimelineDetail,Occupancy,MemoryDetail python ifa_v5_case.py
# msprof op --aic-metrics=Source python ifa_v5_case.py

# --sys-pid-profiling=on
# msprof --ai-core=on --sys-hardware-mem=on --sys-cpu-profiling=on --sys-profiling=on --aic-metrics=ArithmeticUtilization python ifa_v5_case.py

# msprof op simulator --soc-version=Ascend910B2 --output=output_data python ifa_v5_case.py # 仿真出来的结果很奇怪