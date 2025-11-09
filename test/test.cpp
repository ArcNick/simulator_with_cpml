#include <cstdio>
#include <cstdlib>
float a[601][601];
int main() {
    FILE *fp = fopen("test.bin", "rb");
    for (int i = 0; i < 601; i++) {
        fread(a[i], sizeof(float), 601, fp);
    }
    fclose(fp);

    for (int i = 0; i < 100; i++) {
        printf("a[%d][%d] = %0.8f\n", 250, i, a[250][i]);
        printf("a[%d][%d] = %0.8f\n", 250, 601 - i - 1, a[250][601 - i - 1]);
    }
}