# 项目介绍
基于 cuda c++ 的交错网格二维 VTI 介质的弹性波有限差分正演，时间二阶精度、空间八阶精度.

边界选用自由边界+卷积完全匹配层(CPML).

# 用法介绍
```bash
nvcc -rdc=true -I include src/*.cu -o bin/main -std=c++17
./bin/main
```
输出二进制文件的波场快照到 output 文件夹，再使用 tools 中的 getImages.py 即可转为 JPG 文件.
# 网格设置
$\sigma_x$ 和 $\sigma_z$ 两个方向均在整网格点定义.

$v_x$ 为 $x$ 方向定义在半网格点， $z$ 方向定义在整网格点.

$v_z$ 为 $x$ 方向定义在整网格点， $z$ 方向定义在半网格点.

$\tau_{xz}$ 的 $x$ 和 $z$ 方向均定义在半网格点.

网格直观结构如图所示
<img width="2943" height="2251" alt="图片1" src="https://github.com/user-attachments/assets/42939ccd-24b7-4354-affd-a9e079219af3" />
