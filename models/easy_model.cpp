#include <cstdio>
#include <cstdlib>
float a[601][601];
int main() {
    int nz = 601, nx = 601;
    FILE *fp = fopen("rho.bin", "wb");
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            if (i < nz / 2) {
                a[i][j] = 2300;
            } else {
                a[i][j] = 2600;
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
                a[i][j] = 3500;
            } else {
                a[i][j] = 4200;
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
                a[i][j] = 2200;
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
                a[i][j] = 0.05;
            } else {
                a[i][j] = 0.35;
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
                a[i][j] = 0.08;
            } else {
                a[i][j] = 0.25;
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
                a[i][j] = 0.12;
            } else {
                a[i][j] = -0.05;
            }
        }
    }
    for (int i = 0; i < nz; i++) {
        fwrite(a[i], sizeof(float), nx, fp);
    }


    fclose(fp);
}
