/*
 * CUDA Parallel Sobel Edge Detection Implementation
 * File: src/cuda_sobel.cu
 * Course: Parallel Computing Case Study
 * 
 * Description:
 * Implements a complete CUDA GPU pipeline for Grayscale + Sobel Edge Detection:
 * 1. CUDA Kernel for Grayscale Conversion
 * 2. CUDA Kernel for 3x3 Sobel Convolution
 * 3. CUDA Event-based timing distinguishing H2D/D2H memory transfers from kernel processing
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>

// Grayscale Conversion Kernel
__global__ void cuda_grayscale_kernel(const unsigned char* d_bgr, unsigned char* d_gray, int width, int height) {
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

// 3x3 Sobel Edge Detection Kernel
__global__ void cuda_sobel_kernel(const unsigned char* d_gray, unsigned char* d_edges, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Handle boundary pixels
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        d_edges[y * width + x] = 0;
        return;
    }

    // 3x3 Sobel Operator Coefficients
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    // Gx Kernel
    // [-1  0  1]
    // [-2  0  2]
    // [-1  0  1]

    // Gy Kernel
    // [-1 -2 -1]
    // [ 0  0  0]
    // [ 1  2  1]

    sum_x += -1.0f * d_gray[(y - 1) * width + (x - 1)] + 1.0f * d_gray[(y - 1) * width + (x + 1)];
    sum_x += -2.0f * d_gray[(y + 0) * width + (x - 1)] + 2.0f * d_gray[(y + 0) * width + (x + 1)];
    sum_x += -1.0f * d_gray[(y + 1) * width + (x - 1)] + 1.0f * d_gray[(y + 1) * width + (x + 1)];

    sum_y += -1.0f * d_gray[(y - 1) * width + (x - 1)] - 2.0f * d_gray[(y - 1) * width + (x + 0)] - 1.0f * d_gray[(y - 1) * width + (x + 1)];
    sum_y +=  1.0f * d_gray[(y + 1) * width + (x - 1)] + 2.0f * d_gray[(y + 1) * width + (x + 0)] + 1.0f * d_gray[(y + 1) * width + (x + 1)];

    float mag = sqrtf(sum_x * sum_x + sum_y * sum_y);
    d_edges[y * width + x] = static_cast<unsigned char>(fminf(255.0f, mag));
}

bool read_raw_rgb(const std::string& filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    std::string header;
    file >> header >> width >> height;
    int max_val;
    file >> max_val;
    file.ignore(256, '\n');

    data.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    return true;
}

bool write_raw_pgm(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input_ppm> <output_gray_pgm> <output_sobel_pgm> [block_dim] [iterations]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string gray_path = argv[2];
    std::string sobel_path = argv[3];
    int block_dim = (argc >= 5) ? std::stoi(argv[4]) : 16;
    int iterations = (argc >= 6) ? std::stoi(argv[5]) : 5;

    int width = 0, height = 0;
    std::vector<unsigned char> h_bgr;
    if (!read_raw_rgb(input_path, h_bgr, width, height)) {
        std::cerr << "Error reading input image: " << input_path << std::endl;
        return 1;
    }

    std::vector<unsigned char> h_gray(width * height);
    std::vector<unsigned char> h_edges(width * height);

    size_t rgb_bytes = width * height * 3 * sizeof(unsigned char);
    size_t gray_bytes = width * height * sizeof(unsigned char);

    unsigned char *d_bgr = nullptr;
    unsigned char *d_gray = nullptr;
    unsigned char *d_edges = nullptr;

    cudaMalloc(&d_bgr, rgb_bytes);
    cudaMalloc(&d_gray, gray_bytes);
    cudaMalloc(&d_edges, gray_bytes);

    dim3 block(block_dim, block_dim);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h, start_total, stop_total;
    cudaEventCreate(&start_h2d); cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h); cudaEventCreate(&stop_d2h);
    cudaEventCreate(&start_total); cudaEventCreate(&stop_total);

    float total_h2d_ms = 0.0f, total_kernel_ms = 0.0f, total_d2h_ms = 0.0f, total_e2e_ms = 0.0f;

    for (int iter = 0; iter < iterations; ++iter) {
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f, e2e_ms = 0.0f;

        cudaEventRecord(start_total);

        // H2D
        cudaEventRecord(start_h2d);
        cudaMemcpy(d_bgr, h_bgr.data(), rgb_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop_h2d);

        // Kernels
        cudaEventRecord(start_kernel);
        cuda_grayscale_kernel<<<grid, block>>>(d_bgr, d_gray, width, height);
        cuda_sobel_kernel<<<grid, block>>>(d_gray, d_edges, width, height);
        cudaEventRecord(stop_kernel);

        // D2H
        cudaEventRecord(start_d2h);
        cudaMemcpy(h_gray.data(), d_gray, gray_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_edges.data(), d_edges, gray_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_d2h);

        cudaEventRecord(stop_total);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
        cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
        cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
        cudaEventElapsedTime(&e2e_ms, start_total, stop_total);

        if (iter > 0) { // Exclude warm-up
            total_h2d_ms += h2d_ms;
            total_kernel_ms += kernel_ms;
            total_d2h_ms += d2h_ms;
            total_e2e_ms += e2e_ms;
        }
    }

    int valid_iters = std::max(1, iterations - 1);
    float avg_h2d = total_h2d_ms / valid_iters;
    float avg_kernel = total_kernel_ms / valid_iters;
    float avg_d2h = total_d2h_ms / valid_iters;
    float avg_e2e = total_e2e_ms / valid_iters;

    std::cout << "=== CUDA GPU EXECUTION RESULTS (Block: " << block_dim << "x" << block_dim << ") ===\n";
    std::cout << "Host-to-Device (H2D) Transfer: " << avg_h2d << " ms\n";
    std::cout << "Kernel Execution Time:        " << avg_kernel << " ms\n";
    std::cout << "Device-to-Host (D2H) Transfer: " << avg_d2h << " ms\n";
    std::cout << "End-to-End Total Time:         " << avg_e2e << " ms\n";

    write_raw_pgm(gray_path, h_gray, width, height);
    write_raw_pgm(sobel_path, h_edges, width, height);

    cudaFree(d_bgr);
    cudaFree(d_gray);
    cudaFree(d_edges);

    return 0;
}
