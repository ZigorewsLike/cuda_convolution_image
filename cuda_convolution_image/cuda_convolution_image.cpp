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
    int N = 20, M = 20;
    int* a = new int[N * M];
    int* b = new int[N * M];
    int* c = new int[N * M];

    for (int i = 0; i < N * M; i++) {
        a[i] = i;
        b[i] = i * 2 - 3;
    }

    c = calcConvolutionCuda(N, M, a, b);

    for (int i = N - 3; i < N; i++) {
        for (int j = M - 3; j < M; j++) {
            printf("%d + %d = %d\n", a[i * M + j], b[i * M + j], c[i * M + j]);
        }
    }

    int s;
    cin >> s;
}