#include <cstdio>
#include <cstdlib>
float a[601][601];
int main() {
    int nz = 601, nx = 601;
    FILE *fp = fopen("rho.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = 2000;
            } else {
                a[i][j] = 2400;
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
                a[i][j] = 4000;
            } else {
                a[i][j] = 5200;
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
                a[i][j] = 2000;
            } else {
                a[i][j] = 2500;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }
    fclose(fp);
}
