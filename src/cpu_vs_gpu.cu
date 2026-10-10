
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define RUNS 5
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void vector_add(const float *a, const float *b,
                           float *c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static void touch_host(const std::vector<float>& a,
                       const std::vector<float>& b,
                       std::vector<float>& c) {
    volatile float sink = 0.0f;
    for (size_t i = 0; i < a.size(); i += 4096)
        sink += a[i] + b[i] + c[i];
    sink += a.back() + b.back() + c.back();
    (void)sink;
}

static void check_cuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t sizes[] = {1000000, 10000000, 50000000};

    FILE *csv = fopen("results/cuda.csv", "w");
    if (!csv) {
        perror("Cannot open results/cuda.csv");
        return 1;
    }

    fprintf(csv, "version,size,threads,time_seconds,correct\n");

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("CUDA GPU: %s\n", prop.name);
    printf("Block size: %d\n", BLOCK_SIZE);

    for (int s = 0; s < 3; ++s) {
        size_t n = sizes[s];
        size_t bytes = n * sizeof(float);

        std::vector<float> a(n), b(n), c(n), reference(n);

        for (size_t i = 0; i < n; ++i) {
            a[i] = (float)(i % 1000) * 0.25f;
            b[i] = (float)(i % 700) * 0.5f;
            reference[i] = a[i] + b[i];
        }

        touch_host(a, b, c);

        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        /* Untimed warm-up */
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        int blocks = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        vector_add<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

        float best_kernel_ms = 1.0e30f;
        float best_total_ms = 1.0e30f;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int r = 0; r < RUNS; ++r) {
            /* Kernel-only timing */
            CUDA_CHECK(cudaEventRecord(start));
            vector_add<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float kernel_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
            if (kernel_ms < best_kernel_ms)
                best_kernel_ms = kernel_ms;

            /* Total timing: transfers + kernel + return transfer */
            CUDA_CHECK(cudaEventRecord(start));
            CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
            vector_add<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
            if (total_ms < best_total_ms)
                best_total_ms = total_ms;
        }

        int correct = 1;
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != reference[i]) {
                correct = 0;
                break;
            }
        }

        double kernel_seconds = best_kernel_ms / 1000.0;
        double total_seconds = best_total_ms / 1000.0;

        printf("CUDA | N=%zu | kernel=%.9f s | total=%.9f s | correct=%s\n",
               n, kernel_seconds, total_seconds,
               correct ? "YES" : "NO");

        fprintf(csv, "cuda_kernel,%zu,%d,%.9f,%s\n",
                n, BLOCK_SIZE, kernel_seconds, correct ? "YES" : "NO");
        fprintf(csv, "cuda_total,%zu,%d,%.9f,%s\n",
                n, BLOCK_SIZE, total_seconds, correct ? "YES" : "NO");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    fclose(csv);
    printf("CSV saved to results/cuda.csv\n");
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
