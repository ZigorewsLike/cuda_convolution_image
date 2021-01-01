#pragma once

int* calcConvolutionCuda(int N, int M, int* img, int* conv, int cN);
int getCudaDev();