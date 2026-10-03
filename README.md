# Parallel Image Processing: Sequential, OpenMP and CUDA Performance Comparison

## Project Overview

This repository contains a comprehensive Parallel Computing case study evaluating high-resolution image processing pipelines across single-threaded CPU (Sequential C++), shared-memory multi-core CPU (OpenMP C++), and massively parallel GPU hardware (NVIDIA CUDA C++).

The image processing pipeline consists of two stages:
1. **RGB to Grayscale Conversion:** $Gray = 0.299 \times R + 0.587 \times G + 0.114 \times B$
2. **Sobel Edge Detection:** 3x3 spatial convolution calculating gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$

---

## Directory Structure

```text
parallel_image_processing/
│
├── parallel_image_processing_case_study.ipynb   # Main Jupyter Notebook case study report
├── README.md                                     # Project instructions and compilation guide
├── create_test_image.py                         # 4K UHD test image generator
├── run_benchmark_pipeline.py                    # Automated compilation and benchmark runner
├── build_notebook.py                            # Jupyter notebook generator script
│
├── data/
│   └── high_resolution_image.jpg                # Input 4K UHD image (3840 x 2160)
│
├── src/
│   ├── sequential.cpp                           # Single-threaded C++ baseline
│   ├── openmp.cpp                               # Shared-memory OpenMP multi-threaded C++
│   ├── grayscale.cu                             # Custom CUDA grayscale kernel
│   └── cuda_sobel.cu                            # Custom CUDA Sobel edge kernel & host driver
│
├── build/                                       # Compiled binaries and temporary PPM/PGM files
│
├── outputs/                                     # Output images and performance graphs
│   ├── sequential_gray.png
│   ├── sequential_edges.png
│   ├── openmp_gray.png
│   ├── openmp_edges.png
│   ├── cuda_gray.png
│   ├── cuda_edges.png
│   ├── chart1_execution_time.png
│   ├── chart2_speedup.png
│   ├── chart3_openmp_threads.png
│   ├── chart4_openmp_speedup.png
│   ├── chart5_cuda_blocksize.png
│   └── chart6_resolution_scaling.png
│
└── results/
    └── benchmark_results.csv                    # Empirical execution metrics
```

---

## Prerequisites & Installation

### 1. Requirements
- **C++ Compiler:** `g++` (GCC 7+ or TDM-GCC-64) or MSVC / Clang
- **Python:** Python 3.8+ with `opencv-python`, `matplotlib`, `numpy`
- **GPU Acceleration (Optional):** NVIDIA GPU + CUDA Toolkit (`nvcc`)

### 2. Install Python Dependencies
```bash
pip install opencv-python matplotlib numpy jupyter
```

---

## Running the Project

### Option 1: Using Jupyter Notebook
Open the notebook in Jupyter Notebook or VS Code:
```bash
jupyter notebook parallel_image_processing_case_study.ipynb
```
Run all cells sequentially. The notebook dynamically detects system hardware, executes native C++ binaries, records empirical benchmark data, and renders interactive graphs and output images.

### Option 2: Command Line Benchmark Pipeline
To compile and execute the benchmark pipeline directly from terminal:
```bash
python run_benchmark_pipeline.py
```
This script will:
1. Compile `src/sequential.cpp` and `src/openmp.cpp` into `build/`.
2. Compile `src/cuda_sobel.cu` using `nvcc` (if CUDA Toolkit is installed).
3. Execute multi-resolution benchmarks across 1080p, 1440p, and 4K resolutions.
4. Export empirical timing results to `results/benchmark_results.csv`.
5. Generate all 6 visualization charts in `outputs/`.

---

## Empirical Benchmark Findings Summary

- **Sequential CPU (Baseline):** 4K UHD processing requires $\sim 46.28\text{ ms}$, limiting throughput to $\approx 21.6\text{ FPS}$.
- **OpenMP Multi-Core CPU:** Scaling thread count from 1 to 16 threads reduces 4K execution time to $\sim 16.37\text{ ms}$ (**$2.83\times$ speedup**).
- **CUDA GPU Acceleration:** Massively parallel GPU threads execute kernels in $\sim 2.50\text{ ms}$ (**$18.5\times$ kernel speedup**), achieving real-time performance ($> 150\text{ FPS}$).

---

## License & Academic Attribution
Created for the **Parallel Computing Case Study**. Suitable for academic submission.
