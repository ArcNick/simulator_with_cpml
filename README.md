# 项目结构
include 头文件，全是.cuh
src 源文件，全是.cu
# 内容介绍
八阶空间精度，二阶时间精度，以及CPML
# 使用说明
```bash
nvcc -rdc=true -g -G -I include src/*.cu -o bin/main_debug -std=c++17
./bin/main_debug
```
然后跑完之后，运行pyscript.py，ai写的小程序，用于把二进制文件转成jpg
然后再用ai写的.exe小程序就能转gif了
