#ifndef UPDATE_CUH
#define UPDATE_CUH

#include "init.cuh"
#include "cpml.cuh"

__global__ void apply_source(
    Grid_Core::View gc, float src, int posx, int posz
);

__global__ void update_stress(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
);

__global__ void update_velocity(
    Grid_Core::View gc, Grid_Model::View gm, 
    Cpml::View pml, float dx, float dz, float dt
);

#endif