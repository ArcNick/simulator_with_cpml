#include "init.cuh"
#include "update.cuh"
#include "output.cuh"
#include "cpml.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <array>
#include <iostream>

int main() {
    // 读取参数
    Params par;
    par.read("models/params.txt");

    int nx = par.nx;
    int nz = par.nz;
    int nt = par.nt;

    // 初始化并读取模型
    Grid_Model gm_readin(nx, nz, HOST_MEM);
    Grid_Model gm_device(nx, nz, DEVICE_MEM);
    std::array<const char *, 6> files = {
        "models/vp.bin",
        "models/vs.bin",
        "models/rho.bin",
        "models/epsilon.bin",
        "models/delta.bin",
        "models/gamma.bin"
    };
    gm_readin.read(files);

    // 初始化PML
    Cpml cpml(par.view(), "models/cpml.txt");

    printf("网格尺寸: %d x %d\n", par.nx, par.nz);
    printf("空间步长: dx = %f, dz = %f\n", par.dx, par.dz);
    printf("时间步长: dt = %f\n", par.dt);
    printf("总时间步: %d\n", par.nt);
    printf("震源频率: %f Hz\n", par.fpeak);
    printf("震源位置: (%d, %d)\n", par.posx, par.posz);

    // 检查CFL条件
    float dt_max = 0.5f * std::min(par.dx, par.dz) / cpml.cp_max;
    printf("CFL: 最大dt = %f, 实际dt = %f\n", dt_max, par.dt);
    if (par.dt > dt_max) {
        printf("不符合CFL条件\n");
        exit(1);
    }

    // 检查PPW条件
    float ppw = 1900.0f / (2.1 * par.fpeak * par.dx);
    printf("PPW: 最小ppw = 7, 实际ppw = %f\n", ppw);
    

    // 迁移CPU上模型到GPU上
    gm_device.memcpy_to_device_from(gm_readin);

    // 初始化核心计算网格
    Grid_Core gc_host(nx, nz, HOST_MEM);
    Grid_Core gc_device(nx, nz, DEVICE_MEM);

    // 计算刚度参数
    gm_device.calc_stiffness();
    cudaDeviceSynchronize();

    // 生成雷克子波
    float *wl = ricker_wave(par.nt, par.dt, par.fpeak);
    
    // 开始模拟
    Snapshot sshot(gc_host);
    for (int it = 0; it < nt; it++) {
        dim3 gridSize((nx + 15) / 16, (nz + 15) / 16);
        dim3 blockSize(16, 16);

        // 应力更新
        update_stress<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt
        );

        // 加入震源
        apply_source<<<1, 1>>>(
            gc_device.view(), wl[it], par.posx, par.posz
        );
        cudaDeviceSynchronize();
        
        // ψ_stress更新
        cpml_update_psi_stress<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt
        );
        cudaDeviceSynchronize();

        // 速度更新
        update_velocity<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt
        );
        cudaDeviceSynchronize();
        
        // ψ_vel更新
        cpml_update_psi_vel<<<gridSize, blockSize>>>(
            gc_device.view(), gm_device.view(), cpml.view(), 
            par.dx, par.dz, par.dt
        );

        // 自由边界
        apply_free_boundary<<<gridSize, blockSize>>>(gc_device.view());

        if (it % 100 == 0) {
            printf("\r%%%0.2f finished.", 1.0f * it / nt * 100);
            fflush(stdout);
        }

        // 输出波场快照
        if (it % par.snapshot == 0) {
            // 先拷贝到host
            gc_host.memcpy_to_host_from(gc_device);

            // 输出二进制文件
            sshot.output(it, par.dt);
        }
    }
    printf("\r%%100.00 finished.\n");
    fflush(stdout);
    
    delete[] wl;
    wl = nullptr;
}