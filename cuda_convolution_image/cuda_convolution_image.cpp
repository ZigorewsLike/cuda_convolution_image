#include <iostream>
#include <stdio.h>
#include "cuda_funcs.cuh"

using namespace std;

int main()
{
    if (getCudaDev() == 0) {
        cout << "Cuda device not found!";
        return -1;
    }
    int N = 5, M = 5, cN = 3;
    int* a = new int[N * M];
    int* b = new int[N * M];
    int* c = new int[N * M];

    for (int i = 0; i < N * M; i++) {
        a[i] = i;
        b[i] = 1;
    }

    c = calcConvolutionCuda(N, M, a, b, cN);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%-5d ", a[i * N + j]);
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
            printf("%-5d ", c[i * N + j]);
        }
        printf("\n");
    }
}