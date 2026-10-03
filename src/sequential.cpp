/*
 * Sequential Image Processing (Grayscale + Sobel Edge Detection)
 * File: src/sequential.cpp
 * Course: Parallel Computing Case Study
 * 
 * Description:
 * Implements a single-threaded CPU sequential image processing pipeline:
 * 1. RGB/BGR to Grayscale conversion using formula: Gray = 0.299*R + 0.587*G + 0.114*B
 * 2. Sobel Edge Detection using 3x3 horizontal (Gx) and vertical (Gy) gradient kernels
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <string>

// Structure to hold timing details
struct BenchmarkResult {
    double grayscale_time_ms;
    double sobel_time_ms;
    double total_time_ms;
};

// Sequential Grayscale Conversion
void sequential_grayscale(const unsigned char* bgr_in, unsigned char* gray_out, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            unsigned char b = bgr_in[idx + 0];
            unsigned char g = bgr_in[idx + 1];
            unsigned char r = bgr_in[idx + 2];
            // Formula: Gray = 0.299*R + 0.587*G + 0.114*B
            double gray_val = 0.299 * r + 0.587 * g + 0.114 * b;
            gray_out[y * width + x] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, gray_val)));
        }
    }
}

// Sequential Sobel Edge Detection
void sequential_sobel(const unsigned char* gray_in, unsigned char* edges_out, int width, int height) {
    // 3x3 Sobel Kernels
    const int Gx[3][3] = {
        {-1,  0,  1},
        {-2,  0,  2},
        {-1,  0,  1}
    };

    const int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Clear border pixels
    for (int y = 0; y < height; ++y) {
        edges_out[y * width] = 0;
        edges_out[y * width + (width - 1)] = 0;
    }
    for (int x = 0; x < width; ++x) {
        edges_out[x] = 0;
        edges_out[(height - 1) * width + x] = 0;
    }

    // Convolution over interior pixels
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    unsigned char pixel = gray_in[(y + ky) * width + (x + kx)];
                    sum_x += pixel * Gx[ky + 1][kx + 1];
                    sum_y += pixel * Gy[ky + 1][kx + 1];
                }
            }

            double magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            edges_out[y * width + x] = static_cast<unsigned char>(std::min(255.0, magnitude));
        }
    }
}

// Read PPM (P6 - BGR/RGB) image file
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

// Write PGM (P5 - Grayscale) image file
bool write_raw_pgm(const std::string& filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input_ppm> <output_gray_pgm> <output_sobel_pgm> [iterations]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string gray_path = argv[2];
    std::string sobel_path = argv[3];
    int iterations = (argc >= 5) ? std::stoi(argv[4]) : 5;

    int width = 0, height = 0;
    std::vector<unsigned char> bgr_data;
    if (!read_raw_rgb(input_path, bgr_data, width, height)) {
        std::cerr << "Error reading input image: " << input_path << std::endl;
        return 1;
    }

    std::cout << "Loaded Image: " << width << "x" << height << " (" << (width * height) << " pixels)\n";

    std::vector<unsigned char> gray_data(width * height);
    std::vector<unsigned char> edges_data(width * height);

    double total_gray_ms = 0.0;
    double total_sobel_ms = 0.0;

    // Benchmark loop
    for (int iter = 0; iter < iterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sequential_grayscale(bgr_data.data(), gray_data.data(), width, height);
        auto t1 = std::chrono::high_resolution_clock::now();
        sequential_sobel(gray_data.data(), edges_data.data(), width, height);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (iter > 0) { // Exclude warm-up run
            total_gray_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_sobel_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
    }

    int valid_iters = std::max(1, iterations - 1);
    double avg_gray_ms = total_gray_ms / valid_iters;
    double avg_sobel_ms = total_sobel_ms / valid_iters;
    double avg_total_ms = avg_gray_ms + avg_sobel_ms;

    std::cout << "=== SEQUENTIAL EXECUTION RESULTS ===\n";
    std::cout << "Grayscale Time: " << avg_gray_ms << " ms\n";
    std::cout << "Sobel Edge Time: " << avg_sobel_ms << " ms\n";
    std::cout << "Total Processing Time: " << avg_total_ms << " ms\n";

    write_raw_pgm(gray_path, gray_data, width, height);
    write_raw_pgm(sobel_path, edges_data, width, height);
    std::cout << "Outputs saved to " << gray_path << " and " << sobel_path << "\n";

    return 0;
}
