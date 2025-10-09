rm -rf build/
mkdir build && cd build     # 在融合算子源码根目录下创建临时目录并进入
# 后面得按这个顺序来 不然代码不生效
cmake .. 
make package -j $(nproc) 2>&1 | tee ../src/transformer/incre_flash_attention/build.log

