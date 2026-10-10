# CPU vs GPU Data Processing — Team 10

**Parallel Computing Lab Evaluation 1**  
**OpenMP + CUDA: Vector Addition Performance Comparison**

## Team Members

| Name | USN |
|---|---|
| Abhijt R H | 01FE25BCI709 |
| Sagar P M | 01FE25BCI712 |
| Chaitra V | 01FE25BCI702 |
| Ishani Goudar | 01FE22BCI009 |

## 1. Objective

Compare vector addition performance using sequential CPU execution in C, CPU parallel execution using OpenMP with 1, 2, and 4 threads, and GPU execution using CUDA.

The operation is:

```text
C[i] = A[i] + B[i]
```

The experiment tests vectors containing 1,000,000, 10,000,000, and 50,000,000 elements. Each implementation checks its output against a reference result.

## 2. Hardware and Software

- **CPU:** AMD Ryzen 7 7435HS, 8 cores / 16 logical processors
- **GPU:** NVIDIA GeForce RTX 2050, 4 GB
- **Operating system:** Windows
- **C/C++ compiler:** Microsoft Visual C++ Build Tools
- **CUDA Toolkit:** 13.4
- **CUDA block size:** 256 threads
- **Graph generation:** Python, NumPy, Matplotlib

Measurements were collected on the same PC. Results can vary with system load, compiler settings, memory behaviour, and GPU transfer overhead.

## 3. Project Structure

```text
PGC-Lab-Evaluation-1/
├── src/
│   ├── sequential.c
│   ├── cpu_openmp.c
│   └── cpu_vs_gpu.cu
├── data/
├── results/
│   ├── sequential.csv
│   ├── openmp.csv
│   ├── cuda.csv
│   └── combined_results.csv
├── graphs/
│   ├── execution_time.png
│   ├── speedup.png
│   └── efficiency.png
├── presentation/
├── report/
├── generate_graphs.py
└── README.md
```

## 4. Build and Run on Windows

Open an **x64 Native Tools Command Prompt for Visual Studio**, then run the commands from the project directory.

### Sequential CPU

```bat
cd /d D:\PGC-Lab-Evaluation-1
cl /O2 src\sequential.c /Fe:sequential.exe
sequential.exe
```

### OpenMP CPU

```bat
cd /d D:\PGC-Lab-Evaluation-1
cl /O2 /openmp src\cpu_openmp.c /Fe:cpu_openmp.exe
cpu_openmp.exe
```

### CUDA GPU

The following command targets the NVIDIA RTX 2050 (compute capability 8.6):

```bat
cd /d D:\PGC-Lab-Evaluation-1
nvcc -O2 -gencode=arch=compute_86,code=sm_86 src\cpu_vs_gpu.cu -o cuda_test.exe
cuda_test.exe
```

### Generate Graphs

After running the three programs and creating their CSV result files:

```bat
C:\Python314\python.exe generate_graphs.py
```

The script combines the measurements and generates execution-time, speedup, and OpenMP-efficiency graphs in the `graphs/` directory.

## 5. Experimental Results

All recorded sequential, OpenMP, and CUDA runs reported `correct=YES` for all tested vector sizes.

### Sequential CPU

| Vector size | Best execution time (seconds) |
|---:|---:|
| 1,000,000 | 0.000427700 |
| 10,000,000 | 0.014209600 |
| 50,000,000 | 0.069701000 |

### OpenMP CPU

| Vector size | Threads | Best execution time (seconds) |
|---:|---:|---:|
| 1,000,000 | 1 | 0.000488400 |
| 1,000,000 | 2 | 0.000386800 |
| 1,000,000 | 4 | 0.000174400 |
| 10,000,000 | 1 | 0.013682100 |
| 10,000,000 | 2 | 0.014355300 |
| 10,000,000 | 4 | 0.014004800 |
| 50,000,000 | 1 | 0.068019900 |
| 50,000,000 | 2 | 0.070011200 |
| 50,000,000 | 4 | 0.073121500 |

### CUDA GPU

| Vector size | Kernel time (seconds) | Total time (seconds) |
|---:|---:|---:|
| 1,000,000 | 0.000352896 | 0.003286624 |
| 10,000,000 | 0.001706208 | 0.038353920 |
| 50,000,000 | 0.007466528 | 0.170057571 |

CUDA kernel time measures GPU kernel execution. Total time includes data transfers and kernel execution as measured by the program.

## 6. Performance Analysis

**Speedup = Sequential CPU time / Comparison time**

**OpenMP efficiency = OpenMP speedup / Number of threads**

### Speedup Comparison

| Vector size | OpenMP (4 threads) | CUDA kernel only | CUDA total |
|---:|---:|---:|---:|
| 1,000,000 | 2.453× | 1.212× | 0.130× |
| 10,000,000 | 1.015× | 8.328× | 0.371× |
| 50,000,000 | 0.953× | 9.337× | 0.410× |

### Observations

- Every implementation passed the recorded correctness check.
- Four-thread OpenMP improved the smallest workload, but did not improve the larger workloads in these measurements.
- CUDA kernel-only execution was faster than the sequential CPU baseline for all three vector sizes.
- CUDA total execution time, including data transfers, was slower than sequential CPU execution for all three sizes.
- Kernel-only time and total GPU time measure different things and should not be treated as interchangeable.
- These findings describe this experimental run and hardware; results may differ on other systems or under different conditions.

## 7. Graphs

The `graphs/` directory contains:
- `execution_time.png` — execution-time comparison
- `speedup.png` — speedup relative to the sequential CPU baseline
- `efficiency.png` — OpenMP efficiency for different thread counts

The script also saves combined measurements to `results/combined_results.csv`.

## 8. Conclusion

This experiment compares sequential CPU execution, OpenMP multithreading, and CUDA GPU execution for vector addition. The results show that GPU kernel execution can be much faster than the sequential CPU baseline, while data-transfer overhead can make end-to-end GPU execution slower. OpenMP performance depends on workload size and threading overhead, so adding threads does not guarantee better performance.
