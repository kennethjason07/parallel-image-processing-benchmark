/*
 * CUDA Parallel Grayscale Conversion Kernel
 * File: src/grayscale.cu
 * Course: Parallel Computing Case Study
 * 
 * Description:
 * Implements a CUDA GPU kernel for RGB/BGR to Grayscale conversion.
 * Each CUDA thread converts a single pixel in parallel across a 2D Grid of 2D Blocks.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

// CUDA Kernel for Grayscale Conversion
// Formula: Gray = 0.299 * R + 0.587 * G + 0.114 * B
__global__ void rgb_to_grayscale_kernel(const unsigned char* d_bgr, unsigned char* d_gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char b = d_bgr[idx + 0];
        unsigned char g = d_bgr[idx + 1];
        unsigned char r = d_bgr[idx + 2];

        float gray_val = 0.299f * r + 0.587f * g + 0.114f * b;
        d_gray[y * width + x] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, gray_val)));
    }
}

// Host Wrapper Function for CUDA Grayscale Benchmark
extern "C" void launch_cuda_grayscale(const unsigned char* h_bgr, unsigned char* h_gray, int width, int height, float& kernel_time_ms, float& total_time_ms, int block_size = 16) {
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    unsigned char *d_bgr = nullptr;
    unsigned char *d_gray = nullptr;

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_total);

    // 1. Device Memory Allocation
    cudaMalloc(&d_bgr, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    // 2. Host to Device Transfer
    cudaMemcpy(d_bgr, h_bgr, rgb_size, cudaMemcpyHostToDevice);

    // 3. Grid and Block Configuration
    dim3 block(block_size, block_size);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 4. Kernel Launch
    cudaEventRecord(start_kernel);
    rgb_to_grayscale_kernel<<<grid, block>>>(d_bgr, d_gray, width, height);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    // 5. Device to Host Transfer
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&total_time_ms, start_total, stop_total);

    // Cleanup
    cudaFree(d_bgr);
    cudaFree(d_gray);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
}
