#include "init.cuh"
#include "differentiate.cuh"
#include "cpml.cuh"
#include <cstdio>
#include <algorithm>
#include <cmath>

Cpml::Cpml(Params::View gpar, const char *file) {
    read(file);
    nx = gpar.nx;
    nz = gpar.nz;
    mem_location = DEVICE_MEM;
    L = thickness * gpar.dx;

    cudaMalloc(&alpha, thickness * sizeof(float));
    cudaMalloc(&damp, thickness * sizeof(float));
    cudaMalloc(&a, thickness * sizeof(float));
    cudaMalloc(&b, thickness * sizeof(float));
    cudaMalloc(&psi_vx_x, (nx - 1) * nz * sizeof(float));
    cudaMalloc(&psi_vx_z, (nx - 1) * nz * sizeof(float));
    cudaMalloc(&psi_vz_x, nx * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_vz_z, nx * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_sx_x, nx * nz * sizeof(float));
    cudaMalloc(&psi_sx_z, nx * nz * sizeof(float));
    cudaMalloc(&psi_sz_x, nx * nz * sizeof(float));
    cudaMalloc(&psi_sz_z, nx * nz * sizeof(float));
    cudaMalloc(&psi_txz_x, (nx - 1) * (nz - 1) * sizeof(float));
    cudaMalloc(&psi_txz_z, (nx - 1) * (nz - 1) * sizeof(float));

    cudaMemset(alpha, 0, thickness * sizeof(float));
    cudaMemset(damp, 0, thickness * sizeof(float));
    cudaMemset(a, 0, thickness * sizeof(float));
    cudaMemset(b, 0, thickness * sizeof(float));
    cudaMemset(psi_vx_x, 0, (nx - 1) * nz * sizeof(float));
    cudaMemset(psi_vx_z, 0, (nx - 1) * nz * sizeof(float));
    cudaMemset(psi_vz_x, 0, nx * (nz - 1) * sizeof(float));
    cudaMemset(psi_vz_z, 0, nx * (nz - 1) * sizeof(float));
    cudaMemset(psi_sx_x, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sx_z, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sz_x, 0, nx * nz * sizeof(float));
    cudaMemset(psi_sz_z, 0, nx * nz * sizeof(float));
    cudaMemset(psi_txz_x, 0, (nx - 1) * (nz - 1) * sizeof(float));
    cudaMemset(psi_txz_z, 0, (nx - 1) * (nz - 1) * sizeof(float));

    init(gpar);
}

Cpml::~Cpml() {
    if (alpha) {
        cudaFree(alpha);
        alpha = nullptr;
    }
    if (damp) {
        cudaFree(damp);
        damp = nullptr;
    }
    if (a) {
        cudaFree(a);
        a = nullptr;
    }
    if (b) {
        cudaFree(b);
        b = nullptr;
    }
    if (psi_vx_x) {
        cudaFree(psi_vx_x);
        psi_vx_x = nullptr;
    }
    if (psi_vx_z) {
        cudaFree(psi_vx_z);
        psi_vx_z = nullptr;
    }
    if (psi_vz_x) {
        cudaFree(psi_vz_x);
        psi_vz_x = nullptr;
    }
    if (psi_vz_z) {
        cudaFree(psi_vz_z);
        psi_vz_z = nullptr;
    }
    if (psi_sx_x) {
        cudaFree(psi_sx_x);
        psi_sx_x = nullptr;
    }
    if (psi_sx_z) {
        cudaFree(psi_sx_z);
        psi_sx_z = nullptr;
    }
    if (psi_sz_x) {
        cudaFree(psi_sz_x);
        psi_sz_x = nullptr;
    }
    if (psi_sz_z) {
        cudaFree(psi_sz_z);
        psi_sz_z = nullptr;
    }
    if (psi_txz_x) {
        cudaFree(psi_txz_x);
        psi_txz_x = nullptr;
    }
    if (psi_txz_z) {
        cudaFree(psi_txz_z);
        psi_txz_z = nullptr;
    }
}

void Cpml::read(const char *file) {
    FILE *fp = fopen(file, "r");
    if (fp == nullptr) {
        printf("Failed to open CPML file %s\n", file);
        exit(1);
    }
    fscanf(fp, "thickness = %d\n", &thickness);
    fscanf(fp, "N = %f\n", &N);
    fscanf(fp, "cp_max = %f\n", &cp_max);
    fscanf(fp, "Rc = %f\n", &Rc);
    fscanf(fp, "kappa = %f\n", &kappa);

    printf("CPML parameters loaded.\n");
    fclose(fp);
}

__global__ void cpml_init_params(
    Cpml::View cpml, float dx, float dz, float dt
) {
    int i = threadIdx.x;
    if (i >= cpml.thickness) return;
    float x = 1.0 * i / cpml.thickness;
    cpml.damp[i] = cpml.damp0 * powf(x, cpml.N);
    cpml.alpha[i] = cpml.alpha0 * (1.0f - x);

    cpml.b[i] = exp(
        -(cpml.damp[i] / cpml.kappa + cpml.alpha[i]) * dt
    );
    cpml.a[i] = cpml.damp[i] * (cpml.b[i] - 1) / (
        cpml.kappa * (cpml.damp[i] + cpml.kappa * cpml.alpha[i])
    );
    
}

void Cpml::init(Params::View gpar) {
    damp0 = -(N + 1) * cp_max * log(Rc) / (2 * kappa * L);
    alpha0 = M_PI * gpar.fpeak;
    cpml_init_params<<<1, thickness>>>(
        view(), gpar.dx, gpar.dz, gpar.dt
    );
}

__device__ int get_cpml_idx_x(
    const int &lx, const int &ix, const int &thickness
) {
    int res = -1;
    int arr[] = {ix, lx - 1 - ix};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = max(res, arr[i]);
        }
    }
    return thickness - 1 - res;
}

__device__ int get_cpml_idx_z(
    const int &lz, const int &iz, const int &thickness
) {
    int res = -1;
    int arr[] = {iz, lz - 1 - iz};
    for (int i = 0; i < 2; i++) {
        if (0 <= arr[i] && arr[i] < thickness) {
            res = max(res, arr[i]);
        }
    }
    return thickness - 1 - res;
}

__global__ void cpml_update_psi_vel(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = pml.nx, nz = pml.nz;
    int thickness = pml.thickness;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    int pml_idx_x_half_x = get_cpml_idx_x(nx - 1, ix, thickness - 1);
    int pml_idx_z_half_z = get_cpml_idx_z(nz - 1, iz, thickness - 1);

    if (pml_idx_x_half_x < thickness - 1) {
        float a_val = pml.a[pml_idx_x_half_x];
        float b_val = pml.b[pml_idx_x_half_x];
        float dvx_dx = Dx_half_8th(gc.vx, ix, iz, nx - 1, dx);
        float dvz_dx = Dx_int_8th(gc.vz, ix, iz, nx, dx);
        
        PVX_X(ix, iz) = b_val * PVX_X(ix, iz) + a_val * dvx_dx;
        PVZ_X(ix, iz) = b_val * PVZ_X(ix, iz) + a_val * dvz_dx;
    }

    if (pml_idx_z_half_z < thickness - 1) {
        float a_val = pml.a[pml_idx_z_half_z];
        float b_val = pml.b[pml_idx_z_half_z];
        float dvz_dz = Dz_half_8th(gc.vz, ix, iz, nx, dz);
        float dvx_dz = Dz_int_8th(gc.vx, ix, iz, nx - 1, dz);

        PVX_Z(ix, iz) = b_val * PVX_Z(ix, iz) + a_val * dvx_dz;
        PVZ_Z(ix, iz) = b_val * PVZ_Z(ix, iz) + a_val * dvz_dz;
    }
}

__global__ void cpml_update_psi_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = pml.nx, nz = pml.nz;
    int thickness = pml.thickness;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    int pml_idx_x_int = get_cpml_idx_x(nx, ix, thickness);
    int pml_idx_z_int = get_cpml_idx_z(nz, iz, thickness);
    int pml_idx_x_half_x = get_cpml_idx_x(nx - 1, ix, thickness - 1);
    int pml_idx_z_half_z = get_cpml_idx_z(nz - 1, iz, thickness - 1);

    if (pml_idx_x_int < thickness) {
        float a_val = pml.a[pml_idx_x_int];
        float b_val = pml.b[pml_idx_x_int];
        float dsx_dx = Dx_int_8th(gc.sx, ix, iz, nx, dx);

        PSX_X(ix, iz) = b_val * PSX_X(ix, iz) + a_val * dsx_dx;
        PSZ_X(ix, iz) = b_val * PSZ_X(ix, iz) + a_val * dsx_dx;
    }
    
    if (pml_idx_z_int < thickness) {
        float a_val = pml.a[pml_idx_z_int];
        float b_val = pml.b[pml_idx_z_int];
        float dsz_dz = Dz_int_8th(gc.sz, ix, iz, nx, dz);

        PSX_Z(ix, iz) = b_val * PSX_Z(ix, iz) + a_val * dsz_dz;
        PSZ_Z(ix, iz) = b_val * PSZ_Z(ix, iz) + a_val * dsz_dz;
    }
    
    if (ix < nx - 1 && iz < nz - 1) {
        if (pml_idx_x_half_x < thickness - 1) {
            float a_val = pml.a[pml_idx_x_half_x];
            float b_val = pml.b[pml_idx_x_half_x];
            float dtxz_dx = Dx_half_8th(gc.txz, ix, iz, nx - 1, dx);

            PTXZ_X(ix, iz) = b_val * PTXZ_X(ix, iz) + a_val * dtxz_dx;
        }

        if (pml_idx_z_half_z < thickness - 1) {
            float a_val = pml.a[pml_idx_z_half_z];
            float b_val = pml.b[pml_idx_z_half_z];
            float dtxz_dz = Dz_half_8th(gc.txz, ix, iz, nx - 1, dz);

            PTXZ_Z(ix, iz) = b_val * PTXZ_Z(ix, iz) + a_val * dtxz_dz;
        }
    }
}