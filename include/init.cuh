#ifndef INIT_CUH
#define INIT_CUH

#include <cuda_runtime.h>
#include <array>

// 行优先存储：iz * nx + ix
// 参数网格
#define VP(ix, iz) ((gm).vp[(iz) * (gm).nx + (ix)])
#define VS(ix, iz) ((gm).vs[(iz) * (gm).nx + (ix)])
#define RHO(ix, iz) ((gm).rho[(iz) * (gm).nx + (ix)])
#define MU(ix, iz) ((gm).mu[(iz) * (gm).nx + (ix)])
#define LMD(ix, iz) ((gm).lmd[(iz) * (gm).nx + (ix)])

// 核心网格
#define VX(ix, iz) ((gc).vx[(iz) * ((gc).nx - 1) + (ix)])
#define VZ(ix, iz) ((gc).vz[(iz) * (gc).nx + (ix)])
#define SX(ix, iz) ((gc).sx[(iz) * (gc).nx + (ix)])
#define SZ(ix, iz) ((gc).sz[(iz) * (gc).nx + (ix)])
#define TXZ(ix, iz) ((gc).txz[(iz) * ((gc).nx - 1) + (ix)])

#define HOST_MEM 0
#define DEVICE_MEM 1

class Params {
public:
    float fpeak;        // 雷克子波频率
    int nx, nz;         // 网格尺寸
    float dx, dz;       // 网格步长
    int nt;             // 模拟时间步数
    float dt;           // 模拟时间步长
    int posx, posz;     // 炮点位置         
    int snapshot;       // 波场快照间隔
    
    struct View {
        float fpeak;
        int nx, nz;
        float dx, dz;
        int nt;
        float dt;
        int posx, posz;
        int snapshot;
    };
    View view() {
        return (View){
            fpeak, nx, nz, dx, dz, nt, dt, posx, posz, snapshot
        };
    }

    Params() : fpeak(0), nx(0), nz(0), dx(0), dz(0), nt(0), dt(0),
               posx(0), posz(0), snapshot(0) {};
    ~Params() = default;    
    void read(const char *file);
};

class Grid_Model {
public:
    float *vp, *vs, *rho, *mu, *lmd;
    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    struct View {
        float *vp, *vs, *rho, *mu, *lmd;
        int nx, nz;
        bool mem_location;
    };
    View view() {
        return (View){vp, vs, rho, mu, lmd, nx, nz, mem_location};
    }

    Grid_Model(int nx, int nz, bool mem_location);
    ~Grid_Model();

    void read(const std::array<const char *, 3> &files);
    void memcpy_to_device_from(const Grid_Model &rhs);
    void calc_lame();
};

class Grid_Core {
public:
    float *vx, *vz, *sx, *sz, *txz;
    int nx, nz;
    bool mem_location;  // 0 : host, 1 : device

    struct View {
        float *vx, *vz, *sx, *sz, *txz;
        int nx, nz;
        bool mem_location;
    };
    View view() {
        return (View){vx, vz, sx, sz, txz, nx, nz, mem_location};
    }
    
    Grid_Core(int nx, int nz, bool mem_location);
    ~Grid_Core();

    void memcpy_to_host_from(const Grid_Core &rhs);
};

__global__ void lame(
    float *vp, float *vs, float *rho, float *mu, float *lmd,
    int nx, int nz, bool mem_location
);

float *ricker_wave(int nt, float dt, float fpeak);

#endif
