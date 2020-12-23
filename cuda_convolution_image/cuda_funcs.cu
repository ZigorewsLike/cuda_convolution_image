#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>

#include "cuda_funcs.cuh"

using namespace std;
using namespace std::chrono;

__global__ void add_vector(int* a, int* b, int* c, int N)
{
    long tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    long tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int l = 0;
    for (int i = 0; i < 1; i++) {
        l += a[tid_x + tid_y * N] + b[tid_x + tid_y * N];
    }
    c[tid_x + tid_y * N] = l;
}

int* calcConvolutionCuda(int N, int M, int* a, int* b) {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    printf("Cuda device count = %i\n", cuda_count);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float g_time = 0.0;

    int* c = new int[N * M];

    if (cuda_count == 0) {
        cout << "Cuda device not found!";
        return NULL;
    }

    cudaDeviceProp info;
    cudaGetDeviceProperties(&info, 0);
    int maxCudaTreads = info.maxThreadsPerBlock;
    int maxCudaBlocks = (N * M + maxCudaTreads - 1) / maxCudaTreads;
    printf("threads: %i blocks: %i\n", maxCudaTreads, maxCudaBlocks);

    int* dev_a;
    int* dev_b;
    int* dev_c;

    cudaMalloc((void**)&dev_a, N * M * sizeof(int));
    cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    cudaMalloc((void**)&dev_c, N * M * sizeof(int));

    cudaMemcpy(dev_a, a, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * M * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    add_vector << <maxCudaBlocks, maxCudaTreads >> > (dev_a, dev_b, dev_c, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    printf(" time (gpu)= %f mm.\n Calc %i elem\n", g_time, N * M);
    return c;
}

int getCudaDev() {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    return cuda_count;
}