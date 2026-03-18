# CUDA Kernels — Basic GPU Programming Examples

This repository contains a collection of small, focused CUDA kernels that demonstrate foundational GPU programming techniques. Each file implements a single operation using clear, minimal CUDA code, making this repo a practical learning resource for anyone exploring how to write and optimize GPU kernels.

The project includes:

- Basic matrix addition
- Naïve matrix multiplication
- Tiled (shared‑memory) matrix multiplication
- 1D convolution
- 2D convolution
- Parallel sum reduction (shared‑memory optimized)

Each kernel is self‑contained and easy to read, making the repo ideal for experimentation, teaching, or building intuition about CUDA.

## 🛠 Building the Samples (CUDA or HIP)

Just type `make`. The Makefile detects which GPU toolchain is available:

- If **nvcc** is installed → it builds using CUDA  
- If **hipcc** is installed → it builds using HIP/ROCm  


## 💻 No GPU? Try the code on Compiler Explorer (Godbolt)

If you don’t have access to an NVIDIA or AMD GPU, you can still explore how these CUDA/HIP kernels compile by using **Compiler Explorer**:

👉 https://godbolt.org/

Just copy and paste any of the `.cpp` files from this repository into the editor and select:

- **nvcc** (for CUDA)
- **hipcc** (for HIP/ROCm)

Compiler Explorer lets you:

- compile the kernels without hardware  
- inspect the generated PTX or AMD GCN ISA  
- compare naive vs tiled implementations  
- experiment with optimization flags like `-O3`, `--ptx`, or GPU architecture targets  

This is a great way to learn how GPU kernels are lowered into real machine instructions even if you don’t have a GPU available.
