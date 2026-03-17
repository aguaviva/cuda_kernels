#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256   // You can change this if you want

// ------------------------------------------------------------
// EMPTY CUDA REDUCTION KERNEL — YOU IMPLEMENT IT
// Shared memory is already passed via dynamic shared memory
// ------------------------------------------------------------
__global__ void reduceKernel(const float* input, float* output, int N)
{
    // Dynamic shared memory (already allocated by launch)
    extern __shared__ float sdata[];

    // reduce grid to block
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    float res = 0;
    for(int i=x;i<N;i+= blockDim.x * gridDim.x)
    {
        res += input[i];
    }

    sdata[threadIdx.x] = res;
    __syncthreads();

    //reduce block to thread
    for (int stride=blockDim.x/2;stride>0;stride>>=1)
    {
        if (threadIdx.x<stride)
            sdata[threadIdx.x] += sdata[threadIdx.x+stride];
        __syncthreads();
    }

    if( threadIdx.x==0)
        output[blockIdx.x] =sdata[0];

}
// ------------------------------------------------------------


float cpuReduce(const std::vector<float>& v)
{
    float sum = 0.f;
    for (float x : v) sum += x;
    return sum;
}

int main() {
    int N = 1024;

    std::vector<float> input(N);
    for (int i = 0; i < N; i++)
        input[i] = (i % 10) + 1;   // simple pattern

    float cpuResult = cpuReduce(input);

    // Allocate GPU memory
    float *dInput, *dOutput;
    cudaMalloc(&dInput, N * sizeof(float));

    // Worst case: one partial result per block
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&dOutput, numBlocks * sizeof(float));

    cudaMemcpy(dInput, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Dynamic shared memory size
    size_t sharedBytes = BLOCK_SIZE * sizeof(float);

    // ------------------------------------------------------------
    // CALL YOUR KERNEL (shared memory passed here)
    // ------------------------------------------------------------
    reduceKernel<<<numBlocks, BLOCK_SIZE, sharedBytes>>>(dInput, dOutput, N);
    cudaDeviceSynchronize();

    // Copy partial results back
    std::vector<float> partial(numBlocks);
    cudaMemcpy(partial.data(), dOutput, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float gpuResult = cpuReduce(partial);

    cudaFree(dInput);
    cudaFree(dOutput);

    // Print results
    std::cout << "CPU result: " << cpuResult << "\n";
    std::cout << "GPU result: " << gpuResult << "\n";

    std::cout << "\nComparison: "
              << (std::fabs(cpuResult - gpuResult) < 1e-4 ? "MATCH" : "MISMATCH")
              << "\n";

    return 0;
}
