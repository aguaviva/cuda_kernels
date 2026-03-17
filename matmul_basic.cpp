#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matmulKernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x<K && y<M)
    {
        float res = 0;
        for(int i=0;i<N;i++)
        {
            res += A[y*N+i] * B[i*K + x];
        }

        C[y*K+x]=res;
    }
}

int main() {
    int M = 3, N = 4, K = 2;

    std::vector<float> A(M*N), B(N*K), C_cpu(M*K), C_gpu(M*K);

    // Fill matrices with simple values
    for (int i = 0; i < M*N; i++) A[i] = i + 1;
    for (int i = 0; i < N*K; i++) B[i] = (i + 1) * 0.5f;

    // CPU multiplication
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float sum = 0.f;
            for (int n = 0; n < N; n++)
                sum += A[m*N + n] * B[n*K + k];
            C_cpu[m*K + k] = sum;
        }

    // Allocate GPU memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*K*sizeof(float));
    cudaMalloc(&dC, M*K*sizeof(float));

    cudaMemcpy(dA, A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    matmulKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaMemcpy(C_gpu.data(), dC, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // Print results
    std::cout << "CPU result:\n";
    for (int i = 0; i < M*K; i++) {
        std::cout << C_cpu[i] << " ";
        if ((i+1)%K == 0) std::cout << "\n";
    }

    std::cout << "\nGPU result:\n";
    for (int i = 0; i < M*K; i++) {
        std::cout << C_gpu[i] << " ";
        if ((i+1)%K == 0) std::cout << "\n";
    }

    // Compare
    bool ok = true;
    for (int i = 0; i < M*K; i++)
        if (std::fabs(C_cpu[i] - C_gpu[i]) > 1e-5)
            ok = false;

    std::cout << "\nComparison: " << (ok ? "MATCH" : "MISMATCH") << "\n";
    return 0;
}
