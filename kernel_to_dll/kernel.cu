#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>

#define DELLEXPORT extern "C" __declspec(dllexport)
#pragma comment(linker, "/STACK:2000000")
#pragma comment(linker, "/HEAP:2000000")

using namespace std;
using namespace std::chrono;

void handleCudaError(cudaError_t cudaERR) {
    if (cudaERR != cudaSuccess) {
        printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
    }
}

__global__ void add_vector(unsigned int* img, int* conv, unsigned int* c, int N, int M, int cN)
{
    long tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid_x < N * M) {
        int sum = 0;
        int d = 0; // d Normal
        for (int i = 0; i < cN * cN; i++) {
            //int img_x = tid_x % N; 
            //int img_y = tid_x / N;
            int x = tid_x % N - cN / 2 + i % cN;
            int y = tid_x / N - cN / 2 + i / cN;
            int thr_x = -5;
            d += conv[i];
            if (x >= -(cN/2) && x <= (N-1)+(cN/2) && y >= -(cN / 2) && y <= (M - 1) + (cN / 2)) {
                if (x <= -1) x = 0;
                else if (x >= N) x = N - 1;
                if (y <= -1) y = 0;
                else if (y >= M) y = M - 1;
                thr_x = x + y * N;
                sum += conv[i] * img[thr_x];
            }
            //if (i == 0)
            //    printf("imgx = %i, tid_x = %i, imgy = %i\n x = %i, tid_x = %i, new_tid_x = %i, y = %i\n Sum = %i img = %i conv = %i\n--\n", img_x, tid_x, img_y, x, tid_x, thr_x, y, sum, img[thr_x], conv[i]);
        }
        if (d != 0) sum /= abs(d);
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        //printf("tid_x = %i Val = %i\n --\n", tid_x, sum);
        c[tid_x] = sum;
    }
}

DELLEXPORT unsigned int* calcConvolutionCuda(int N, int M, unsigned int* img, int* conv, int cN) {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    //printf("\n ---------------------- \nCuda device count = %i\n", cuda_count);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float g_time = 0.0;

    unsigned int* c = new unsigned int[N * M];

    if (cuda_count == 0) {
        //cout << "Cuda device not found!";
        return NULL;
    }
    cudaDeviceProp info;
    cudaGetDeviceProperties(&info, 0);
    unsigned int maxCudaTreads = info.maxThreadsPerBlock;
    if (maxCudaTreads > N * M) {
        maxCudaTreads = N * M;
    }
    unsigned int maxCudaBlocks = (N * M + maxCudaTreads - 1) / maxCudaTreads;

    //printf("Init %i \n", img[0]);

    //printf("N = %i M = %i\n threads: %i blocks: %i\n", N, M, maxCudaTreads, maxCudaBlocks);

    unsigned int* dev_a;
    int* dev_b;
    unsigned int* dev_c;

    //printf("CudaMalloc \n");

    cudaMalloc((void**)&dev_a, N * M * sizeof(unsigned int));
    cudaMalloc((void**)&dev_b, N * M * sizeof(int));
    cudaMalloc((void**)&dev_c, N * M * sizeof(unsigned int));

    //printf("CudaMemcpy \n");

    handleCudaError(cudaMemcpy(dev_a, img, N * M * sizeof(unsigned int), cudaMemcpyHostToDevice));
    handleCudaError(cudaMemcpy(dev_b, conv, N * M * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start, 0);

    //printf("Cuda run \n");

    add_vector << < maxCudaTreads, maxCudaBlocks >> > (dev_a, dev_b, dev_c, N, M, cN);

    //printf("cudaMemcpy \n");

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&g_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    //printf("OK \n");

    //printf(" time (gpu)= %f mm.\n Calc %i elem\n", g_time, N * M);
    return c;
}

DELLEXPORT int getCudaDev() {
    int cuda_count;
    cudaGetDeviceCount(&cuda_count);
    return cuda_count;
}

