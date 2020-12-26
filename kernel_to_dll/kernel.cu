#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>

#define DELLEXPORT extern "C" __declspec(dllexport)

using namespace std;
using namespace std::chrono;

__global__ void add_vector(int* img, int* conv, int* c, int N, int M, int cN)
{
    long tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid_x < N * M) {
        int sum = 0;
        int d = 0; // d Normal
        for (int i = 0; i < cN * cN; i++) {
            int img_x = tid_x % N;
            int img_y = tid_x / N;
            int x = tid_x % N - cN / 2 + i % cN;
            int y = tid_x / N - cN / 2 + i / cN;
            int thr_x = -5;
            if (x >= 0 && x < N && y >= 0 && y < N) {
                d += conv[i];
                thr_x = x + y * N;
                sum += conv[i] * img[thr_x];
            }
            //if (i == 1)
                //printf("imgx = %i, tid_x = %i, imgy = %i\n x = %i, tid_x = %i, new_tid_x = %i, y = %i\n --\n", img_x, tid_x, img_y, x, tid_x, thr_x, y);
        }
        sum /= d;
        c[tid_x] = sum;
    }
}

DELLEXPORT int* calcConvolutionCuda(int N, int M, int* img, int* conv, int cN) {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    //printf("Cuda device count = %i\n", cuda_count);

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
    unsigned int maxCudaTreads = info.maxThreadsPerBlock;
    if (maxCudaTreads > N * M) {
        maxCudaTreads = N * M;
    }
    unsigned int maxCudaBlocks = (N * M + maxCudaTreads - 1) / maxCudaTreads;

    //printf("threads: %i blocks: %i\n", maxCudaTreads, maxCudaBlocks);

    int* dev_a;
    int* dev_b;
    int* dev_c;

    cudaMalloc((void**)&dev_a, N * M * sizeof(int));
    cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    cudaMalloc((void**)&dev_c, N * M * sizeof(int));

    cudaMemcpy(dev_a, img, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, conv, N * M * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    add_vector << < maxCudaTreads, maxCudaBlocks >> > (dev_a, dev_b, dev_c, N, M, cN);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    //printf(" time (gpu)= %f mm.\n Calc %i elem\n", g_time, N * M);
    return c;
}

DELLEXPORT int getCudaDev() {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    return cuda_count;
}