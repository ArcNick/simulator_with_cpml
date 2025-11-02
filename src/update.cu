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

    float lmd = LMD(ix, iz);
    float mu_int = MU(ix, iz);
    
    float dvx_dx = Dx_half_8th(gc.vx, ix, iz, nx - 1, dx);
    float dvz_dz = Dz_half_8th(gc.vz, ix, iz, nx, dz);
    float dvz_dx = Dx_int_8th(gc.vz, ix, iz, nx, dx);
    float dvx_dz = Dz_int_8th(gc.vx, ix, iz, nx - 1, dz);
    
    if (get_cpml_idx_x(nx - 1, ix, pml.thickness) < pml.thickness) {
        dvx_dx = dvx_dx / pml.kappa + PVX_X(ix, iz);
        dvx_dz = dvx_dz / pml.kappa + PVX_Z(ix, iz);
    }
    if (get_cpml_idx_z(nz - 1, iz, pml.thickness) < pml.thickness) {
        dvz_dz = dvz_dz / pml.kappa + PVZ_Z(ix, iz);
        dvz_dx = dvz_dx / pml.kappa + PVZ_X(ix, iz);
    }

    SX(ix, iz) += dt * (
        (lmd + 2.0f * mu_int) * dvx_dx + lmd * dvz_dz
    );
    SZ(ix, iz) += dt * (
        lmd * dvx_dx + (lmd + 2.0f * mu_int) * dvz_dz
    );

    // txz : (nx-1) × (nz-1)
    if (ix < nx - 1 && iz < nz - 1) {
        float mu_half = (MU(ix, iz) + MU(ix + 1, iz)) * 0.5f;
        TXZ(ix, iz) += dt * mu_half * (
            dvz_dx + dvx_dz
        );
    }
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

    // vx : (nx-1) × nz
    if (ix < nx - 1) {
        float rho_half_x = (RHO(ix, iz) + RHO(ix + 1, iz)) * 0.5f;
        float dtxz_dz = Dz_half_8th(gc.txz, ix, iz, nx - 1, dz);
        float dsx_dx = Dx_int_8th(gc.sx, ix, iz, nx, dx);

        if (get_cpml_idx_x(nx, ix, pml.thickness) < pml.thickness) {
            dsx_dx = dsx_dx / pml.kappa + PSX_X(ix, iz);
        }
        if (get_cpml_idx_z(nz - 1, iz, pml.thickness) < pml.thickness) {
            dtxz_dz = dtxz_dz / pml.kappa + PTXZ_Z(ix, iz);
        }

        VX(ix, iz) += dt / rho_half_x * (dsx_dx + dtxz_dz);
    }
    
    // vz : nx × (nz-1)
    if (iz < nz - 1) {
        float rho_half_z = (RHO(ix, iz) + RHO(ix, iz + 1)) * 0.5f;
        float dtxz_dx = Dx_half_8th(gc.txz, ix, iz, nx - 1, dx);
        float dsz_dz = Dz_int_8th(gc.sz, ix, iz, nx, dz);

        if (get_cpml_idx_x(nx - 1, ix, pml.thickness) < pml.thickness) {
            dtxz_dx = dtxz_dx / pml.kappa + PTXZ_X(ix, iz);
        }
        if (get_cpml_idx_z(nz, iz, pml.thickness) < pml.thickness) {
            dsz_dz = dsz_dz / pml.kappa + PSZ_Z(ix, iz);
        }

        VZ(ix, iz) += dt / rho_half_z * (dtxz_dx + dsz_dz);
    }
}