
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

// ------------------------------------------------------------
// EMPTY CUDA MATMUL KERNEL — YOU IMPLEMENT IT
// Shared memory is already passed (dynamic or static)
// ------------------------------------------------------------
__global__ void matmulKernel_non_tiled(const float* A,
                             const float* B,
                             float* C,
                             int M, int N, int K)
{
    extern __shared__ float smem[];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x<K && y<M)
    {
        float res = 0;
        for(int i=0;i<N;i++)
        {
            res += A[y*N + i] * B[i*K + x];
        }
        C[y*K+x] = res;
    }
}

// ------------------------------------------------------------
// EMPTY CUDA MATMUL KERNEL — YOU IMPLEMENT IT
// Shared memory is already passed (dynamic or static)
// ------------------------------------------------------------
__global__ void matmulKernel(const float* A,
                             const float* B,
                             float* C,
                             int M, int N, int K)
{
    extern __shared__ float smem[];
    float *tA = smem;
    float *tB = smem + TILE*TILE;

    //int x = blockDim.x * blockIdx.x + threadIdx.x;
    //int y = blockDim.y * blockIdx.y + threadIdx.y;

    int xt = blockIdx.x * TILE + threadIdx.x;
    int yt = blockIdx.y * TILE + threadIdx.y;


    float res = 0.0;

    for (int t=0;t<(N+TILE)/TILE;t++)
    {
        int xtt = t*TILE + threadIdx.x;
        int ytt = t*TILE + threadIdx.y;


        if (xtt<N && yt<M)
            tA[TILE*threadIdx.y + threadIdx.x] = A[yt*N + xtt];
        else 
            tA[TILE*threadIdx.y + threadIdx.x] = 0;

        if (xt<K && ytt<N)
            tB[TILE*threadIdx.y + threadIdx.x] = B[ytt*K + xt];
        else
            tB[TILE*threadIdx.y + threadIdx.x] = 0;

        __syncthreads();

        for(int j=0;j<TILE;j++)
        {
            res += tA[TILE*threadIdx.y + j] *tB[TILE*j + threadIdx.x];
        }

    }
    
    if (xt<K && yt<M)
        C[yt*K+xt] = res;
}

// ------------------------------------------------------------


void cpuMatmul(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C,
               int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float sum = 0.f;
            for (int n = 0; n < N; n++)
                sum += A[m*N + n] * B[n*K + k];
            C[m*K + k] = sum;
        }
}

int main() {
    int M = 19, N = 5, K = 18;

    std::vector<float> A(M*N), B(N*K), C_cpu(M*K), C_gpu(M*K);

    // Fill matrices with simple values
    for (int i = 0; i < M*N; i++) A[i] = i + 1;
    for (int i = 0; i < N*K; i++) B[i] = (i + 1) * 0.1f;

    // CPU reference
    cpuMatmul(A, B, C_cpu, M, N, K);

    // Allocate GPU memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*K*sizeof(float));
    cudaMalloc(&dC, M*K*sizeof(float));

    cudaMemcpy(dA, A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N*K*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Dynamic shared memory size (2 tiles: A and B)
    size_t sharedBytes = 2 * TILE * TILE * sizeof(float);

    // ------------------------------------------------------------
    // CALL YOUR KERNEL (shared memory passed here)
    // ------------------------------------------------------------
    matmulKernel<<<grid, block, sharedBytes>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C_gpu.data(), dC, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Print CPU result
    std::cout << "CPU result:\n";
    for (int i = 0; i < M*K; i++) {
        std::cout << C_cpu[i] << " ";
        if ((i+1)%K == 0) std::cout << "\n";
    }

    // Print GPU result
    std::cout << "\nGPU result:\n";
    for (int i = 0; i < M*K; i++) {
        std::cout << C_gpu[i] << " ";
        if ((i+1)%K == 0) std::cout << "\n";
    }

    // Compare
    bool ok = true;
    for (int i = 0; i < M*K; i++)
        if (std::fabs(C_cpu[i] - C_gpu[i]) > 1e-2)
            ok = false;

    std::cout << "\nComparison: " << (ok ? "MATCH" : "MISMATCH") << "\n";
    return 0;
}
