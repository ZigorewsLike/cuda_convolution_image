#include <iostream>
#include <stdio.h>
#include "cuda_funcs.cuh"

using namespace std;

unsigned char* readBMP()
{
    int i;
    FILE* f = fopen("F:\\Pict\\1.bmp", "rb");
    unsigned char info[54];

    // read the 54-byte header
    fread(info, sizeof(unsigned char), 54, f);

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    // allocate 3 bytes per pixel
    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size];

    // read the rest of the data at once
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);

    for (i = 0; i < size; i += 3)
    {
        // flip the order of every 3 bytes
        unsigned char tmp = data[i];
        data[i] = data[i + 2];
        data[i + 2] = tmp;
    }

    return data;
}

int main()
{
    if (getCudaDev() == 0) {
        cout << "Cuda device not found!";
        return -1;
    }

    FILE* f = fopen("F:\\Pict\\2.bmp", "rb");
    unsigned char info[54];

    fread(info, sizeof(unsigned char), 54, f);

    int N = *(int*)&info[18];
    int M = *(int*)&info[22];

    int size = 3 * N * M;
    unsigned char* data = new unsigned char[size];

    fread(data, sizeof(unsigned char), size, f);
    fclose(f);

    for (int i = 0; i < size; i += 3) {
        int tmp = data[i];
        data[i] = data[i + 2];
        data[i + 2] = tmp;
    }
    printf("%i %i %i \n", data[0], data[1], data[2]);

    //int N = 5, M = 5, cN = 3;
    int cN = 3;
    int* a = new int[3 * N * M];
    int* b = new int[N * M];
    int* c = new int[N * M];

    for (int i = 0; i < 3 * N * M; i++) {
        printf("%c %i", data[i], data[i]);
        a[i] = (int)data[i];
    }
    for (int i = 0; i < cN*cN; i++) {
        b[i] = 0;
    }
    b[4] = 1;

    c = calcConvolutionCuda(N, M, a, b, cN);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("[");
            for (int k = 0; k < 3; k++) {
                printf("%-3d ", a[(i * N + j)*3 + k]);
            }
            printf("] ");
        }
        printf("\n");
    }
    printf("-----------\n");
    for (int i = 0; i < cN; i++) {
        for (int j = 0; j < cN; j++) {
            printf("%-5d ", b[i * cN + j]);
        }
        printf("\n");
    }
    printf("-----------\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("[");
            for (int k = 0; k < 3; k++) {
                printf("%-3d ", c[(i * N + j) * 3 + k]);
            }
            printf("] ");
        }
        printf("\n");
    }
}