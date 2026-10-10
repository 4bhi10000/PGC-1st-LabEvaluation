
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <float.h>

#define RUNS 5

static double now_seconds(void) {
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    if (frequency.QuadPart == 0)
        QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

static void touch_arrays(const float *a, const float *b,
                         const float *c, size_t n) {
    volatile float sink = 0.0f;

    for (size_t i = 0; i < n; i += 4096)
        sink += a[i] + b[i] + c[i];

    sink += a[n - 1] + b[n - 1] + c[n - 1];
    (void)sink;
}

static void add_vectors(const float *a, const float *b,
                        float *c, size_t n) {
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

int main(void) {
    const size_t sizes[] = {1000000, 10000000, 50000000};

    FILE *csv = fopen("results/sequential.csv", "w");
    if (!csv) {
        perror("Cannot open results/sequential.csv");
        return 1;
    }

    fprintf(csv, "version,size,threads,time_seconds,correct\n");

    for (int s = 0; s < 3; ++s) {
        size_t n = sizes[s];

        float *a = (float *)malloc(n * sizeof(float));
        float *b = (float *)malloc(n * sizeof(float));
        float *c = (float *)malloc(n * sizeof(float));
        float *reference = (float *)malloc(n * sizeof(float));

        if (!a || !b || !c || !reference) {
            fprintf(stderr, "Memory allocation failed for N=%zu\n", n);
            free(a);
            free(b);
            free(c);
            free(reference);
            fclose(csv);
            return 1;
        }

        for (size_t i = 0; i < n; ++i) {
            a[i] = (float)(i % 1000) * 0.25f;
            b[i] = (float)(i % 700) * 0.5f;
        }

        touch_arrays(a, b, c, n);

        /* Untimed warm-up and reference result */
        add_vectors(a, b, reference, n);

        double best = DBL_MAX;

        for (int r = 0; r < RUNS; ++r) {
            double start = now_seconds();
            add_vectors(a, b, c, n);
            double elapsed = now_seconds() - start;

            if (elapsed < best)
                best = elapsed;
        }

        int correct = 1;
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != reference[i]) {
                correct = 0;
                break;
            }
        }

        printf("Sequential | N=%zu | best=%.9f s | correct=%s\n",
               n, best, correct ? "YES" : "NO");

        fprintf(csv, "sequential,%zu,1,%.9f,%s\n",
                n, best, correct ? "YES" : "NO");

        free(a);
        free(b);
        free(c);
        free(reference);
    }

    fclose(csv);
    printf("CSV saved to results/sequential.csv\n");
    return 0;
}
