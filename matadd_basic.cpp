#include <iostream>
#include <vector>
#include <cmath>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C,
                                  int rows, int cols)
{
    // TODO: implement your GPU matrix addition here
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x<rows*cols)
    {
        C[x] = A[x]+B[x];
    }
}

void matrix_add_cpu(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int rows, int cols)
{
    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int rows = 4;
    int cols = 6;
    int size = rows * cols;

    std::vector<float> A(size);
    std::vector<float> B(size);
    std::vector<float> cpu_C(size);
    std::vector<float> gpu_C(size);

    // Fill matrices with simple values
    for (int i = 0; i < size; i++) {
        A[i] = i + 1;
        B[i] = (i + 1) * 10;
    }

    // CPU version
    matrix_add_cpu(A, B, cpu_C, rows, cols);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 128;
    int blocks = (size + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, rows, cols);

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(gpu_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "CPU result:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::cout << cpu_C[i * cols + j] << " ";
        std::cout << "\n";
    }

    std::cout << "\nGPU result:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::cout << gpu_C[i * cols + j] << " ";
        std::cout << "\n";
    }

    // Compare
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (std::fabs(cpu_C[i] - gpu_C[i]) > 1e-4f) {
            match = false;
            break;
        }
    }

    std::cout << "\n" << (match ? "Results match\n" : "Results DO NOT match\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
