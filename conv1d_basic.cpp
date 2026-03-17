#include <iostream>
#include <vector>
#include <cmath>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size)
{
    for (int i=0;i<(input_size-kernel_size+1);i++)
    {
        float res=0;
        for (int k=0;k<kernel_size;k++)
        {
            res += input[i+k] * kernel[k];
        }
        output[i]=res;
    }
}

void convolution_1d_cpu(const std::vector<float>& input,
                        const std::vector<float>& kernel,
                        std::vector<float>& output)
{
    int input_size = input.size();
    int kernel_size = kernel.size();
    int output_size = input_size - kernel_size + 1;

    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += input[i + k] * kernel[k];
        }
        output[i] = sum;
    }
}

int main()
{
    int input_size = 16;
    int kernel_size = 5;
    int output_size = input_size - kernel_size + 1;

    std::vector<float> input(input_size);
    std::vector<float> kernel(kernel_size);
    std::vector<float> cpu_output(output_size);
    std::vector<float> gpu_output(output_size);

    // Fill input and kernel with simple values
    for (int i = 0; i < input_size; i++) input[i] = i + 1;
    for (int i = 0; i < kernel_size; i++) kernel[i] = 1.0f / kernel_size;

    // CPU version
    convolution_1d_cpu(input, kernel, cpu_output);

    // Allocate GPU memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 128;
    int blocks = (output_size + threads - 1) / threads;
    convolution_1d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output,
                                               input_size, kernel_size);

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(gpu_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "CPU output:\n";
    for (float v : cpu_output) std::cout << v << " ";
    std::cout << "\n\nGPU output:\n";
    for (float v : gpu_output) std::cout << v << " ";
    std::cout << "\n\n";

    // Compare
    bool match = true;
    for (int i = 0; i < output_size; i++) {
        if (std::fabs(cpu_output[i] - gpu_output[i]) > 1e-4f) {
            match = false;
            break;
        }
    }

    std::cout << (match ? "Results match\n" : "Results DO NOT match\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
