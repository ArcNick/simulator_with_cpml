#include "init.cuh"
#include <cmath>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <cuda_runtime.h>

void Params::read(const char *file) {
    FILE *fp = fopen(file, "r");
    if (fp == nullptr) {
        printf("Failed to open parameter file %s\n", file);
        exit(1);
    }

    fscanf(fp, "fpeak = %f\n", &fpeak);
    fscanf(fp, "nx = %d\n", &nx);
    fscanf(fp, "nz = %d\n", &nz);
    fscanf(fp, "dx = %f\n", &dx);
    fscanf(fp, "dz = %f\n", &dz);
    fscanf(fp, "nt = %d\n", &nt);
    fscanf(fp, "dt = %f\n", &dt);
    fscanf(fp, "posx = %d\n", &posx);
    fscanf(fp, "posz = %d\n", &posz);
    fscanf(fp, "snapshot = %d\n", &snapshot);

    printf("Parameters loaded.\n");
    fclose(fp);
}

__global__ void lame(
    float *vp, float *vs, float *rho, float *mu, float *lmd,
    int nx, int nz, bool mem_location
) {
    if (mem_location != DEVICE_MEM) {
        printf("RE in \"calc_lame\"!\n");
        return;
    }
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iz < nz) {
        int idx = iz * nx + ix;
        lmd[idx] = rho[idx] * (vp[idx] * vp[idx] - 2 * vs[idx] * vs[idx]);
        mu[idx] = rho[idx] * vs[idx] * vs[idx];
    }
}

Grid_Model::Grid_Model(int nx, int nz, bool mem_location) 
    : nx(nx), nz(nz), mem_location(mem_location) {
    if (mem_location == HOST_MEM) {
        vp = new float[nx * nz]();
        vs = new float[nx * nz]();
        rho = new float[nx * nz]();
        mu = new float[nx * nz]();
        lmd = new float[nx * nz]();
    } else {
        cudaMalloc((void**)&vp, nx * nz * sizeof(float));
        cudaMalloc((void**)&vs, nx * nz * sizeof(float));
        cudaMalloc((void**)&rho, nx * nz * sizeof(float));
        cudaMalloc((void**)&mu, nx * nz * sizeof(float));
        cudaMalloc((void**)&lmd, nx * nz * sizeof(float));
        cudaMemset(vp, 0, nx * nz * sizeof(float));
        cudaMemset(vs, 0, nx * nz * sizeof(float));
        cudaMemset(rho, 0, nx * nz * sizeof(float));
        cudaMemset(mu, 0, nx * nz * sizeof(float));
        cudaMemset(lmd, 0, nx * nz * sizeof(float));
    }
}

Grid_Model::~Grid_Model() {
    if (mem_location == HOST_MEM) {
        if (vp) {
            delete[] vp;
            vp = nullptr;
        }
        if (vs) {
            delete[] vs;
            vs = nullptr;
        }
        if (rho) {
            delete[] rho;
            rho = nullptr;
        }
        if (mu) {
            delete[] mu;
            mu = nullptr;
        }
        if (lmd) {
            delete[] lmd;
            lmd = nullptr;
        }
    } else {
        if (vp) {
            cudaFree(vp);
            vp = nullptr;
        }
        if (vs) {
            cudaFree(vs);
            vs = nullptr;
        }
        if (rho) {
            cudaFree(rho);
            rho = nullptr;
        }
        if (mu) {
            cudaFree(mu);
            mu = nullptr;
        }
        if (lmd) {
            cudaFree(lmd);
            lmd = nullptr;
        }
    }
}

void Grid_Model::read(const std::array<const char *, 3> &files) {
    if (mem_location == DEVICE_MEM) {
        printf("RE in \"Grid_Model::read\"!\n");
        exit(1);
    }

    std::array<float *, 3> dst = {vp, vs, rho};
    FILE *fp = nullptr;
    for (int i = 0; i < 3; i++) {
        fp = fopen(files[i], "rb");
        for (int iz = 0; iz < nz; iz++) {
            fread(dst[i] + iz * nx, sizeof(float), nx, fp);
        }
        fclose(fp);
    }
}

void Grid_Model::memcpy_to_device_from(const Grid_Model &rhs) {
    if (rhs.mem_location == mem_location || mem_location == HOST_MEM) {
        printf("RE in \"Grid_Model::memcpy_to_device_from\"!\n");
        exit(1);
    }

    int total_bytes = nx * nz * sizeof(float);
    cudaMemcpy(vp, rhs.vp, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vs, rhs.vs, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(rho, rhs.rho, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mu, rhs.mu, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(lmd, rhs.lmd, total_bytes, cudaMemcpyHostToDevice);
}

void Grid_Model::calc_lame() {
    dim3 gridSize((nx + 15) / 16, (nz + 15) / 16);
    dim3 blockSize(16, 16);
    lame<<<gridSize, blockSize>>>(
        vp, vs, rho, mu, lmd, nx, nz, mem_location
    );
}

Grid_Core::Grid_Core(int nx, int nz, bool mem_location) 
    : nx(nx), nz(nz), mem_location(mem_location) {
    if (mem_location == HOST_MEM) {
        vx = new float[(nx - 1) * nz]();
        vz = new float[nx * (nz - 1)]();
        sx = new float[nx * nz]();
        sz = new float[nx * nz]();
        txz = new float[(nx - 1) * (nz - 1)]();
    } else {
        cudaMalloc((void**)&vx, (nx - 1) * nz * sizeof(float));
        cudaMalloc((void**)&vz, nx * (nz - 1) * sizeof(float));
        cudaMalloc((void**)&sx, nx * nz * sizeof(float));
        cudaMalloc((void**)&sz, nx * nz * sizeof(float));
        cudaMalloc((void**)&txz, (nx - 1) * (nz - 1) * sizeof(float));

        cudaMemset(vx, 0, (nx - 1) * nz * sizeof(float));
        cudaMemset(vz, 0, nx * (nz - 1) * sizeof(float));
        cudaMemset(sx, 0, nx * nz * sizeof(float));
        cudaMemset(sz, 0, nx * nz * sizeof(float));
        cudaMemset(txz, 0, (nx - 1) * (nz - 1) * sizeof(float));
    }
}

Grid_Core::~Grid_Core() {
    if (mem_location == HOST_MEM) {
        if (vx) {
            delete[] vx;
            vx = nullptr;
        }
        if (vz) {
            delete[] vz;
            vz = nullptr;
        }
        if (sx) {
            delete[] sx;
            sx = nullptr;
        }
        if (sz) {
            delete[] sz;
            sz = nullptr;
        }
        if (txz) {
            delete[] txz;
            txz = nullptr;
        }
    } else {
        if (vx) {
            cudaFree(vx);
            vx = nullptr;
        }
        if (vz) {
            cudaFree(vz);
            vz = nullptr;
        }
        if (sx) {
            cudaFree(sx);
            sx = nullptr;
        }
        if (sz) {
            cudaFree(sz);
            sz = nullptr;
        }
        if (txz) {
            cudaFree(txz);
            txz = nullptr;
        }
    }
}

void Grid_Core::memcpy_to_host_from(const Grid_Core &rhs) {
    if (rhs.mem_location == mem_location || rhs.mem_location == HOST_MEM) {
        printf("RE in \"Grid_Core::memcpy_to_host_from\"!\n");
        exit(1);
    }
    cudaMemcpy(vx, rhs.vx, (nx - 1) * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz, rhs.vz, nx * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sx, rhs.sx, nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sz, rhs.sz, nx * nz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(txz, rhs.txz, (nx - 1) * (nz - 1) * sizeof(float), cudaMemcpyDeviceToHost);
}

float *ricker_wave(int nt, float dt, float fpeak) {
    float *wavelet = new float[nt]();
    float T = 1.0 / fpeak;
    for (int it = 0; it < nt; it++) {
        float t = it * dt - T;
        float temp = M_PI * fpeak * t;
        temp *= temp;
        wavelet[it] = (1 - 2 * temp) * exp(-temp);
    }
    return wavelet;
}