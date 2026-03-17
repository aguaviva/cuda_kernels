#include <iostream>
#include <vector>
#include <cmath>

__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int in_rows, int in_cols,
                                      int k_rows, int k_cols)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int out_rows = in_rows-k_rows+1;
    int out_cols = in_cols-k_cols+1;

    if ( (x<out_cols) && (y<out_rows) )
    {
        float res = 0;

        for (int j=0;j<k_rows;j++)
        {
            for (int i=0;i<k_cols;i++)
            {
                res += input[(y+j)*in_cols + (i+x)] * kernel[j*k_cols + i];
            }    
        }    
        output[y*out_cols+x] = res;
    }
}

void convolution_2d_cpu(const std::vector<float>& input,
                        const std::vector<float>& kernel,
                        std::vector<float>& output,
                        int in_rows, int in_cols,
                        int k_rows, int k_cols)
{
    int out_rows = in_rows - k_rows + 1;
    int out_cols = in_cols - k_cols + 1;

    for (int r = 0; r < out_rows; r++) {
        for (int c = 0; c < out_cols; c++) {

            float sum = 0.0f;

            for (int kr = 0; kr < k_rows; kr++) {
                for (int kc = 0; kc < k_cols; kc++) {
                    int in_r = r + kr;
                    int in_c = c + kc;
                    sum += input[in_r * in_cols + in_c] *
                           kernel[kr * k_cols + kc];
                }
            }

            output[r * out_cols + c] = sum;
        }
    }
}

int main()
{
    int in_rows = 6;
    int in_cols = 8;

    int k_rows = 3;
    int k_cols = 3;

    int out_rows = in_rows - k_rows + 1;
    int out_cols = in_cols - k_cols + 1;

    int in_size = in_rows * in_cols;
    int k_size = k_rows * k_cols;
    int out_size = out_rows * out_cols;

    std::vector<float> input(in_size);
    std::vector<float> kernel(k_size);
    std::vector<float> cpu_output(out_size);
    std::vector<float> gpu_output(out_size);

    // Fill input and kernel with simple values
    for (int i = 0; i < in_size; i++) input[i] = i + 1;
    for (int i = 0; i < k_size; i++) kernel[i] = 1.0f / k_size;

    // CPU version
    convolution_2d_cpu(input, kernel, cpu_output,
                       in_rows, in_cols, k_rows, k_cols);

    // Allocate GPU memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, in_size * sizeof(float));
    cudaMalloc(&d_kernel, k_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), k_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y);

    convolution_2d_kernel<<<grid, block>>>(d_input, d_kernel, d_output,
                                               in_rows, in_cols,
                                               k_rows, k_cols);

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(gpu_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "CPU output:\n";
    for (int r = 0; r < out_rows; r++) {
        for (int c = 0; c < out_cols; c++)
            std::cout << cpu_output[r * out_cols + c] << " ";
        std::cout << "\n";
    }

    std::cout << "\nGPU output:\n";
    for (int r = 0; r < out_rows; r++) {
        for (int c = 0; c < out_cols; c++)
            std::cout << gpu_output[r * out_cols + c] << " ";
        std::cout << "\n";
    }

    // Compare
    bool match = true;
    for (int i = 0; i < out_size; i++) {
        if (std::fabs(cpu_output[i] - gpu_output[i]) > 1e-4f) {
            match = false;
            break;
        }
    }

    std::cout << "\n" << (match ? "Results match\n" : "Results DO NOT match\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
