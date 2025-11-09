#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

float a[601][601];

std::unordered_map<std::string, std::pair<float, float>> params;
int main() {
    int nz = 601, nx = 601;

    params["vp0"] = {3600, 3600};
    params["vs0"] = {1900, 1900};
    params["rho"] = {2000, 2000};
    params["epsilon"] = {0, 0};
    params["delta"] = {0, 0};
    params["gamma"] = {0, 0};

    FILE *fp = fopen("rho.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["rho"].first;
            } else {
                a[i][j] = params["rho"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }
    
    fp = fopen("vp.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["vp0"].first;
            } else {
                a[i][j] = params["vp0"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }
    
    fp = fopen("vs.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["vs0"].first;
            } else {
                a[i][j] = params["vs0"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }

    fp = fopen("gamma.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["gamma"].first;
            } else {
                a[i][j] = params["gamma"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }

    fp = fopen("epsilon.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["epsilon"].first;
            } else {
                a[i][j] = params["epsilon"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }


    fp = fopen("delta.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = params["delta"].first;
            } else {
                a[i][j] = params["delta"].second;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }

    fclose(fp);
}
