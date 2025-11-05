#ifndef OUTPUT_CUH
#define OUTPUT_CUH

#include "init.cuh"
#include <vector>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// 打包波场快照的信息
struct Array_Info {
    float* data;
    int lenx, lenz;
    std::string name;
    Array_Info(float* d, int lx, int lz, const std::string &n) 
        : data(d), lenx(lx), lenz(lz), name(n) {}
    ~Array_Info() = default;
};

// 控制波场快照输出的类
class Snapshot {
public:
    int nz, nx;
    std::vector<Array_Info> arrays;
    fs::path output_dir;

    Snapshot(const Grid_Core &g);
    ~Snapshot() = default;

    void output(int it, float dt);
};

#endif