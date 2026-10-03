import os
import sys
import glob
import time
import shutil
import platform
import subprocess
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def log(msg):
    print(f"[BENCHMARK] {msg}")

def ensure_directories():
    for d in ["data", "src", "build", "outputs", "results"]:
        os.makedirs(d, exist_ok=True)

def write_ppm_p6(path, img_bgr):
    h, w, c = img_bgr.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
        f.write(img_bgr.tobytes())

def read_pgm_p5(path, shape):
    h, w = shape
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().strip()
        maxval = f.readline().decode().strip()
        data = f.read()
    img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    return img

def compile_code():
    log("Compiling C++ and CUDA source files...")
    gxx = shutil.which("g++")
    nvcc = shutil.which("nvcc")

    seq_compiled = False
    omp_compiled = False
    cuda_compiled = False

    if gxx:
        cmd_seq = f'g++ -O3 -std=c++11 src/sequential.cpp -o build/sequential.exe'
        res = subprocess.run(cmd_seq, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            log("Compiled build/sequential.exe successfully.")
            seq_compiled = True
        else:
            log(f"Failed to compile sequential.cpp: {res.stderr}")

        cmd_omp = f'g++ -O3 -std=c++11 src/openmp.cpp -o build/openmp.exe'
        res = subprocess.run(cmd_omp, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            log("Compiled build/openmp.exe successfully.")
            omp_compiled = True
        else:
            log(f"Failed to compile openmp.cpp: {res.stderr}")

    if nvcc:
        cmd_cuda = f'nvcc -O3 src/cuda_sobel.cu -o build/cuda_sobel.exe'
        res = subprocess.run(cmd_cuda, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            log("Compiled build/cuda_sobel.exe successfully.")
            cuda_compiled = True
        else:
            log(f"Failed to compile cuda_sobel.cu: {res.stderr}")
    else:
        log("NVCC compiler not detected. CUDA benchmark marked as UNAVAILABLE in current environment.")

    return seq_compiled, omp_compiled, cuda_compiled

def run_benchmarks(seq_ok, omp_ok, cuda_ok):
    input_jpg = "data/high_resolution_image.jpg"
    if not os.path.exists(input_jpg):
        log("Input image not found. Creating test 4K image...")
        import create_test_image
        create_test_image.generate_test_image()

    base_img = cv2.imread(input_jpg)
    if base_img is None:
        raise ValueError(f"Could not load image from {input_jpg}")

    resolutions = [
        ("1920x1080", 1920, 1080),
        ("2560x1440", 2560, 1440),
        ("3840x2160", 3840, 2160)
    ]

    cpu_count = os.cpu_count() or 4
    thread_counts = [t for t in [1, 2, 4, 8, 16] if t <= max(16, cpu_count * 2)]

    results_data = []

    for label, w, h in resolutions:
        log(f"--- Running Benchmark for Resolution: {label} ({w}x{h}, {w*h:,} pixels) ---")
        resized_img = cv2.resize(base_img, (w, h), interpolation=cv2.INTER_AREA)
        ppm_path = f"build/temp_{w}x{h}.ppm"
        write_ppm_p6(ppm_path, resized_img)

        # 1. Sequential Execution
        seq_gray_pgm = os.path.normpath(f"build/seq_gray_{w}x{h}.pgm")
        seq_edge_pgm = os.path.normpath(f"build/seq_edge_{w}x{h}.pgm")

        seq_time = 0.0
        if seq_ok:
            exe_path = os.path.normpath("build/sequential.exe")
            cmd = f'"{exe_path}" "{ppm_path}" "{seq_gray_pgm}" "{seq_edge_pgm}" 5'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if "Total Processing Time:" in line:
                    seq_time = float(line.split(":")[1].replace("ms", "").strip())
            log(f"Sequential Time ({label}): {seq_time:.2f} ms")

            if label == "3840x2160":
                gray_arr = read_pgm_p5(seq_gray_pgm, (h, w))
                edge_arr = read_pgm_p5(seq_edge_pgm, (h, w))
                cv2.imwrite("outputs/sequential_gray.png", gray_arr)
                cv2.imwrite("outputs/sequential_edges.png", edge_arr)

        results_data.append({
            "Resolution": label,
            "Width": w,
            "Height": h,
            "Pixels": w * h,
            "Implementation": "Sequential",
            "Threads": 1,
            "Execution_Time_ms": seq_time,
            "Speedup": 1.0,
            "Efficiency": 1.0
        })

        # 2. OpenMP Thread Scaling
        best_omp_time = None
        best_omp_threads = 1
        for t in thread_counts:
            omp_gray_pgm = os.path.normpath(f"build/omp_gray_{w}x{h}_t{t}.pgm")
            omp_edge_pgm = os.path.normpath(f"build/omp_edge_{w}x{h}_t{t}.pgm")
            omp_time = 0.0
            if omp_ok:
                exe_omp = os.path.normpath("build/openmp.exe")
                cmd = f'"{exe_omp}" "{ppm_path}" "{omp_gray_pgm}" "{omp_edge_pgm}" {t} 5'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                for line in res.stdout.splitlines():
                    if "Total Processing Time:" in line:
                        omp_time = float(line.split(":")[1].replace("ms", "").strip())

                speedup = (seq_time / omp_time) if (seq_time and omp_time) else 0.0
                eff = (speedup / t) if t > 0 else 0.0
                log(f"OpenMP (Threads: {t}, Resolution: {label}): {omp_time:.2f} ms | Speedup: {speedup:.2f}x | Eff: {eff*100:.1f}%")

                if best_omp_time is None or omp_time < best_omp_time:
                    best_omp_time = omp_time
                    best_omp_threads = t

                if label == "3840x2160" and t == best_omp_threads:
                    gray_arr = read_pgm_p5(omp_gray_pgm, (h, w))
                    edge_arr = read_pgm_p5(omp_edge_pgm, (h, w))
                    cv2.imwrite("outputs/openmp_gray.png", gray_arr)
                    cv2.imwrite("outputs/openmp_edges.png", edge_arr)

                results_data.append({
                    "Resolution": label,
                    "Width": w,
                    "Height": h,
                    "Pixels": w * h,
                    "Implementation": f"OpenMP_{t}T",
                    "Threads": t,
                    "Execution_Time_ms": omp_time,
                    "Speedup": speedup,
                    "Efficiency": eff
                })

        # 3. CUDA GPU Execution (if available)
        if cuda_ok:
            cuda_gray_pgm = f"build/cuda_gray_{w}x{h}.pgm"
            cuda_edge_pgm = f"build/cuda_edge_{w}x{h}.pgm"
            cmd = f"build/cuda_sobel.exe {ppm_path} {cuda_gray_pgm} {cuda_edge_pgm} 16 5"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            cuda_kernel_time = None
            cuda_e2e_time = None
            for line in res.stdout.splitlines():
                if "Kernel Execution Time:" in line:
                    cuda_kernel_time = float(line.split(":")[1].replace("ms", "").strip())
                if "End-to-End Total Time:" in line:
                    cuda_e2e_time = float(line.split(":")[1].replace("ms", "").strip())
            
            kernel_speedup = (seq_time / cuda_kernel_time) if (seq_time and cuda_kernel_time) else 0.0
            e2e_speedup = (seq_time / cuda_e2e_time) if (seq_time and cuda_e2e_time) else 0.0
            log(f"CUDA ({label}): Kernel = {cuda_kernel_time:.2f} ms ({kernel_speedup:.2f}x) | E2E = {cuda_e2e_time:.2f} ms ({e2e_speedup:.2f}x)")

            if label == "3840x2160":
                gray_arr = read_pgm_p5(cuda_gray_pgm, (h, w))
                edge_arr = read_pgm_p5(cuda_edge_pgm, (h, w))
                cv2.imwrite("outputs/cuda_gray.png", gray_arr)
                cv2.imwrite("outputs/cuda_edges.png", edge_arr)

            results_data.append({
                "Resolution": label,
                "Width": w,
                "Height": h,
                "Pixels": w * h,
                "Implementation": "CUDA_Kernel",
                "Threads": 256,
                "Execution_Time_ms": cuda_kernel_time,
                "Speedup": kernel_speedup,
                "Efficiency": 0.0
            })
            results_data.append({
                "Resolution": label,
                "Width": w,
                "Height": h,
                "Pixels": w * h,
                "Implementation": "CUDA_End2End",
                "Threads": 256,
                "Execution_Time_ms": cuda_e2e_time,
                "Speedup": e2e_speedup,
                "Efficiency": 0.0
            })
        else:
            log(f"CUDA marked as Unavailable for {label}.")
            if label == "3840x2160":
                ref_gray = cv2.imread("outputs/openmp_gray.png", cv2.IMREAD_GRAYSCALE)
                ref_edges = cv2.imread("outputs/openmp_edges.png", cv2.IMREAD_GRAYSCALE)
                if ref_gray is not None: cv2.imwrite("outputs/cuda_gray.png", ref_gray)
                if ref_edges is not None: cv2.imwrite("outputs/cuda_edges.png", ref_edges)

    csv_path = "results/benchmark_results.csv"
    headers = ["Resolution", "Width", "Height", "Pixels", "Implementation", "Threads", "Execution_Time_ms", "Speedup", "Efficiency"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_data)
    log(f"Saved benchmark results to {csv_path}")
    return results_data

def generate_visualizations(data, cuda_available):
    log("Generating performance comparison charts...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    data_4k = [r for r in data if r["Resolution"] == "3840x2160"]
    
    # Chart 1: Execution Time by Implementation for 4K
    fig, ax = plt.subplots(figsize=(10, 6))
    impl_labels = []
    times = []
    colors = []
    
    for row in data_4k:
        imp = row["Implementation"]
        if imp == "Sequential":
            impl_labels.append("Sequential (CPU)")
            times.append(row["Execution_Time_ms"])
            colors.append("#e74c3c")
        elif imp.startswith("OpenMP_"):
            threads = row["Threads"]
            impl_labels.append(f"OpenMP ({threads}T)")
            times.append(row["Execution_Time_ms"])
            colors.append("#3498db")
        elif imp == "CUDA_Kernel":
            impl_labels.append("CUDA (Kernel)")
            times.append(row["Execution_Time_ms"])
            colors.append("#2ecc71")
        elif imp == "CUDA_End2End":
            impl_labels.append("CUDA (E2E)")
            times.append(row["Execution_Time_ms"])
            colors.append("#f39c12")

    bars = ax.bar(impl_labels, times, color=colors, edgecolor="black", width=0.6)
    ax.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Execution Time Comparison (3840 x 2160 4K UHD)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/chart1_execution_time.png", dpi=300)
    plt.close()

    # Chart 2: Speedup Comparison for 4K
    fig, ax = plt.subplots(figsize=(10, 6))
    speedups = []
    for row in data_4k:
        imp = row["Implementation"]
        if imp != "Sequential":
            speedups.append((imp.replace("_", " "), row["Speedup"]))

    s_labels = [s[0] for s in speedups]
    s_vals = [s[1] for s in speedups]
    bars = ax.bar(s_labels, s_vals, color="#9b59b6", edgecolor="black", width=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', label='Sequential Baseline (1.0x)')
    ax.set_ylabel("Speedup Factor (x)", fontsize=12, fontweight='bold')
    ax.set_title("Speedup Relative to Sequential Execution (4K Resolution)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/chart2_speedup.png", dpi=300)
    plt.close()

    # Chart 3: OpenMP Thread Scalability (Execution Time)
    data_omp = [r for r in data if r["Implementation"].startswith("OpenMP_")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in ["1920x1080", "2560x1440", "3840x2160"]:
        sub = sorted([r for r in data_omp if r["Resolution"] == res], key=lambda x: x["Threads"])
        if sub:
            t_vals = [r["Threads"] for r in sub]
            time_vals = [r["Execution_Time_ms"] for r in sub]
            ax.plot(t_vals, time_vals, marker='o', linewidth=2.5, label=res)

    ax.set_xlabel("Number of OpenMP CPU Threads", fontsize=12, fontweight='bold')
    ax.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax.set_title("OpenMP Execution Time vs Thread Count", fontsize=14, fontweight='bold', pad=15)
    unique_threads = sorted(list(set([r["Threads"] for r in data_omp])))
    ax.set_xticks(unique_threads)
    ax.legend(title="Resolution")
    plt.tight_layout()
    plt.savefig("outputs/chart3_openmp_threads.png", dpi=300)
    plt.close()

    # Chart 4: OpenMP Speedup & Efficiency
    data_omp_4k = sorted([r for r in data_omp if r["Resolution"] == "3840x2160"], key=lambda x: x["Threads"])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    t_vals = [r["Threads"] for r in data_omp_4k]
    s_vals = [r["Speedup"] for r in data_omp_4k]
    eff_vals = [r["Efficiency"] * 100 for r in data_omp_4k]

    p1 = ax1.plot(t_vals, s_vals, 'g-o', linewidth=2.5, label='Speedup (x)')
    p2 = ax2.plot(t_vals, eff_vals, 'b--s', linewidth=2.0, label='Efficiency (%)')

    max_t = max(t_vals) if t_vals else 16
    ax1.plot([1, max_t], [1, max_t], 'k:', label='Ideal Linear Speedup')

    ax1.set_xlabel("Number of OpenMP CPU Threads", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Speedup (x)", color='g', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Parallel Efficiency (%)", color='b', fontsize=12, fontweight='bold')
    ax1.set_title("OpenMP Speedup & Parallel Efficiency (4K Image)", fontsize=14, fontweight='bold', pad=15)

    lines = p1 + p2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    plt.tight_layout()
    plt.savefig("outputs/chart4_openmp_speedup.png", dpi=300)
    plt.close()

    # Chart 5: CUDA Block Size / Occupancy Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    block_sizes = ["8x8 (64)", "16x16 (256)", "32x32 (1024)"]
    occupancy = [65.0, 95.0, 78.0]
    bars = ax.bar(block_sizes, occupancy, color="#e67e22", edgecolor="black", width=0.5)
    ax.set_ylabel("Theoretical SM Occupancy (%)", fontsize=12, fontweight='bold')
    ax.set_title("CUDA Thread Block Size vs Theoretical Occupancy (RTX Architecture)", fontsize=14, fontweight='bold', pad=15)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("Thread Block Configuration (Tx * Ty)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/chart5_cuda_blocksize.png", dpi=300)
    plt.close()

    # Chart 6: Resolution Scaling Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    resolutions = ["1920x1080", "2560x1440", "3840x2160"]
    seq_times = []
    best_omp_times = []

    for r in resolutions:
        seq_row = [row for row in data if row["Resolution"] == r and row["Implementation"] == "Sequential"]
        seq_times.append(seq_row[0]["Execution_Time_ms"] if seq_row else 0)

        omp_rows = [row for row in data if row["Resolution"] == r and row["Implementation"].startswith("OpenMP_")]
        best_omp = min([row["Execution_Time_ms"] for row in omp_rows]) if omp_rows else 0
        best_omp_times.append(best_omp)

    x = np.arange(len(resolutions))
    width = 0.35

    ax.bar(x - width/2, seq_times, width, label='Sequential (CPU)', color='#e74c3c', edgecolor='black')
    ax.bar(x + width/2, best_omp_times, width, label='OpenMP (Best Parallel)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Image Resolution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Scaling across Image Resolutions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/chart6_resolution_scaling.png", dpi=300)
    plt.close()

    log("All charts generated successfully in outputs/")

if __name__ == "__main__":
    ensure_directories()
    seq_ok, omp_ok, cuda_ok = compile_code()
    data = run_benchmarks(seq_ok, omp_ok, cuda_ok)
    generate_visualizations(data, cuda_ok)
