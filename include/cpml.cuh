#ifndef CPML_CUH
#define CPML_CUH

#include <init.cuh>
#include <cmath>

#define PVX_X(ix, iz) (pml.psi_vx_x[(iz) * (pml.nx - 1) + (ix)])
#define PVX_Z(ix, iz) (pml.psi_vx_z[(iz) * (pml.nx - 1) + (ix)])
#define PVZ_X(ix, iz) (pml.psi_vz_x[(iz) * pml.nx + (ix)])
#define PVZ_Z(ix, iz) (pml.psi_vz_z[(iz) * pml.nx + (ix)])
#define PSX_X(ix, iz) (pml.psi_sx_x[(iz) * pml.nx + (ix)])
#define PSX_Z(ix, iz) (pml.psi_sx_z[(iz) * pml.nx + (ix)])
#define PSZ_X(ix, iz) (pml.psi_sz_x[(iz) * pml.nx + (ix)])
#define PSZ_Z(ix, iz) (pml.psi_sz_z[(iz) * pml.nx + (ix)])
#define PTXZ_X(ix, iz) (pml.psi_txz_x[(iz) * (pml.nx - 1) + (ix)])
#define PTXZ_Z(ix, iz) (pml.psi_txz_z[(iz) * (pml.nx - 1) + (ix)])

class Cpml {
public:
    int thickness;                  // CPML层厚度（网格点数）    
    float N;                        // 阻尼剖面指数
    float cp_max;                   // 最大纵波波速
    float L;                        // CPML层厚度
    float Rc;                       // 反射系数
    float damp0;                    // 最大阻尼系数
    float alpha0;                   // 最大频移因子
    float kappa;                    // 拉伸因子

    float *alpha;                   // 频移因子
    float *damp;                    // 阻尼系数
    float *a;                       // 卷积系数
    float *b;                       // 衰减系数

    float *psi_vx_x, *psi_vx_z;     // 记忆变量
    float *psi_vz_x, *psi_vz_z;
    float *psi_sx_x, *psi_sx_z;
    float *psi_sz_x, *psi_sz_z;
    float *psi_txz_x, *psi_txz_z;

    int nx, nz;
    bool mem_location;

    struct View {
        int thickness;
        float N;
        float cp_max;
        float L;
        float Rc;
        float damp0;
        float alpha0;
        float kappa;

        float *alpha;
        float *damp;
        float *a;
        float *b;

        float *psi_vx_x, *psi_vx_z;
        float *psi_vz_x, *psi_vz_z;
        float *psi_sx_x, *psi_sx_z;
        float *psi_sz_x, *psi_sz_z;
        float *psi_txz_x, *psi_txz_z;

        int nx, nz;
        bool mem_location;
    };

    View view() {
        return (View){
            thickness, N, cp_max, L, Rc, 
            damp0, alpha0, kappa,
            alpha, damp, a, b,
            psi_vx_x, psi_vx_z,
            psi_vz_x, psi_vz_z,
            psi_sx_x, psi_sx_z,
            psi_sz_x, psi_sz_z,
            psi_txz_x, psi_txz_z,
            nx, nz, mem_location
        };
    }

    Cpml(Params::View gpar, const char *file);
    ~Cpml();

private:
    void read(const char *file);
    void init(Params::View gpar);
};

__global__ void cpml_init_params(
    Cpml::View cpml, float dx, float dz, float dt
);

__device__ int get_cpml_idx_x(
    const int &lx, const int &ix, const int &thickness
);

__device__ int get_cpml_idx_z(
    const int &lz, const int &iz, const int &thickness
);

__global__ void cpml_update_psi_vel(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View cpml, float dx, float dz, float dt
);

__global__ void cpml_update_psi_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View cpml, float dx, float dz, float dt
);

#endif