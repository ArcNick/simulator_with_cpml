#include "init.cuh"
#include "differentiate.cuh"
#include "update.cuh"
#include "cpml.cuh"
#include <cstdio>

__global__ void apply_source(
    Grid_Core::View gc, float src, int posx, int posz
) {
    gc.sx[posz * gc.nx + posx] += src;
    gc.sz[posz * gc.nx + posx] += src;
}

__global__ void update_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = gc.nx, nz = gc.nz;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    float C11 = C11(ix, iz);
    float C13 = C13(ix, iz);
    float C33 = C33(ix, iz);
    // float C44 = C44(ix, iz);
    
    float dvx_dx = Dx_half_8th(gc.vx, ix, iz, nx - 1, dx);
    float dvz_dz = Dz_half_8th(gc.vz, ix, iz, nx, dz);
    float dvz_dx = Dx_int_8th(gc.vz, ix, iz, nx, dx);
    float dvx_dz = Dz_int_8th(gc.vx, ix, iz, nx - 1, dz);
    
    int pml_idx_x_int = get_cpml_idx_x(nx, ix, pml.thickness);
    int pml_idx_z_int = get_cpml_idx_z(nz, iz, pml.thickness);
    int pml_idx_x_half = get_cpml_idx_x(nx - 1, ix, pml.thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z(nz - 1, iz, pml.thickness - 1);

    // if (pml_idx_x_half < pml.thickness - 1) {
    //     dvx_dx = dvx_dx / pml.kappa[pml_idx_x_half] + PVX_X(ix, iz);
    //     dvx_dz = dvx_dz / pml.kappa[pml_idx_x_half] + PVX_Z(ix, iz);
    // }
    // if (pml_idx_z_half < pml.thickness - 1) {
    //     dvz_dz = dvz_dz / pml.kappa[pml_idx_z_half] + PVZ_Z(ix, iz);
    //     dvz_dx = dvz_dx / pml.kappa[pml_idx_z_half] + PVZ_X(ix, iz);
    // }

    if (pml_idx_x_int < pml.thickness) {
        dvz_dx = dvz_dx / pml.kappa_int[pml_idx_x_int] + PVZ_X(ix, iz);
    }
    if (pml_idx_z_int < pml.thickness) {
        dvx_dz = dvx_dz / pml.kappa_int[pml_idx_z_int] + PVX_Z(ix, iz);
    }
    if (pml_idx_x_half < pml.thickness - 1) {
        dvx_dx = dvx_dx / pml.kappa_half[pml_idx_x_half] + PVX_X(ix, iz);
    }
    if (pml_idx_z_half < pml.thickness - 1) {
        dvz_dz = dvz_dz / pml.kappa_half[pml_idx_z_half] + PVZ_Z(ix, iz);
    }

    SX(ix, iz) += dt * (C11 * dvx_dx + C13 * dvz_dz);
    SZ(ix, iz) += dt * (C13 * dvx_dx + C33 * dvz_dz);

    // txz : (nx-1) × (nz-1)
    
    float C44_half = 0.25 * (
        C44(ix, iz) + C44(ix + 1, iz) + 
        C44(ix, iz + 1) + C44(ix + 1, iz + 1)
    );
    TXZ(ix, iz) += dt * C44_half * (dvz_dx + dvx_dz);
}

__global__ void update_velocity(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = gc.nx, nz = gc.nz;

    if (ix < 4 || ix >= nx - 4 || iz < 4 || iz >= nz - 4) {
        return;
    }

    int pml_idx_x_int = get_cpml_idx_x(nx, ix, pml.thickness);
    int pml_idx_z_int = get_cpml_idx_z(nz, iz, pml.thickness);
    int pml_idx_x_half = get_cpml_idx_x(nx - 1, ix, pml.thickness - 1);
    int pml_idx_z_half = get_cpml_idx_z(nz - 1, iz, pml.thickness - 1);

    // vx : (nx-1) × nz
    float rho_half_x = (RHO(ix, iz) + RHO(ix + 1, iz)) * 0.5f;
    float dtxz_dz = Dz_half_8th(gc.txz, ix, iz, nx - 1, dz);
    float dsx_dx = Dx_int_8th(gc.sx, ix, iz, nx, dx);

    if (pml_idx_x_int < pml.thickness) {
        dsx_dx = dsx_dx / pml.kappa_int[pml_idx_x_int] + PSX_X(ix, iz);
    }
    if (pml_idx_z_half < pml.thickness - 1) {
        dtxz_dz = dtxz_dz / pml.kappa_half[pml_idx_z_half] + PTXZ_Z(ix, iz);
    }
    VX(ix, iz) += dt / rho_half_x * (dsx_dx + dtxz_dz);
    
    // vz : nx × (nz-1)
    float rho_half_z = (RHO(ix, iz) + RHO(ix, iz + 1)) * 0.5f;
    float dtxz_dx = Dx_half_8th(gc.txz, ix, iz, nx - 1, dx);
    float dsz_dz = Dz_int_8th(gc.sz, ix, iz, nx, dz);

    if (pml_idx_x_half < pml.thickness - 1) {
        dtxz_dx = dtxz_dx / pml.kappa_half[pml_idx_x_half] + PTXZ_X(ix, iz);
    }
    if (pml_idx_z_int < pml.thickness) {
        dsz_dz = dsz_dz / pml.kappa_int[pml_idx_z_int] + PSZ_Z(ix, iz);
    }
    VZ(ix, iz) += dt / rho_half_z * (dtxz_dx + dsz_dz);

    // 在update_stress或update_velocity中添加
if (ix == 10 && iz == 10) {  // 选择一个测试点
    int pml_idx_left = get_cpml_idx_x(nx, 4, pml.thickness);  // 左边界附近
    int pml_idx_right = get_cpml_idx_x(nx, nx-5, pml.thickness);  // 右边界附近
    
    printf("左边界索引: %d, 右边界索引: %d\n", pml_idx_left, pml_idx_right);
    printf("左参数: a=%f, b=%f, 右参数: a=%f, b=%f\n",
           pml.a_half[pml_idx_left], pml.b_half[pml_idx_left],
           pml.a_half[pml_idx_right], pml.b_half[pml_idx_right]);
}

    // Dirichlet
    if (ix == 4 || iz == 4 || ix == nx - 5 || iz == nz - 5) {
        VX(ix, iz) = VZ(ix, iz) = 0;
    }
}