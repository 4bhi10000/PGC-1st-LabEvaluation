
import csv
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
GRAPHS = os.path.join(ROOT, "graphs")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(GRAPHS, exist_ok=True)

FILES = [
    os.path.join(RESULTS, "sequential.csv"),
    os.path.join(RESULTS, "openmp.csv"),
    os.path.join(RESULTS, "cuda.csv"),
]

# Load actual measurements from all three programs.
rows = []
for path in FILES:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding program first."
        )
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows.extend(csv.DictReader(f))

measurements = {}
for row in rows:
    version = row["version"].strip()
    size = int(row["size"])
    threads = int(row["threads"])
    seconds = float(row["time_seconds"])
    correct = row["correct"].strip().upper()

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid time in {row}")

    measurements[(version, size, threads)] = {
        "time": seconds,
        "correct": correct,
    }

sizes = [1_000_000, 10_000_000, 50_000_000]
thread_counts = [1, 2, 4]

def get_measurement(version, size, threads):
    key = (version, size, threads)
    if key not in measurements:
        raise ValueError(f"Missing measurement: {key}")
    return measurements[key]

# Sequential time is the baseline for each size.
serial = {
    n: get_measurement("sequential", n, 1)["time"]
    for n in sizes
}

# Calculate speedup and CPU-thread efficiency.
combined = []
for n in sizes:
    base = serial[n]
    for version, threads in [
        ("sequential", 1),
        ("openmp", 1),
        ("openmp", 2),
        ("openmp", 4),
        ("cuda_kernel", 256),
        ("cuda_total", 256),
    ]:
        m = get_measurement(version, n, threads)
        speedup = base / m["time"]

        efficiency = (
            speedup / threads if version == "openmp"
            else (1.0 if version == "sequential" else None)
        )

        combined.append({
            "version": version,
            "size": n,
            "threads": threads,
            "time_seconds": m["time"],
            "time_ms": m["time"] * 1000,
            "speedup": speedup,
            "efficiency": efficiency,
            "correct": m["correct"],
        })

combined_path = os.path.join(RESULTS, "combined_results.csv")
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    fields = [
        "version", "size", "threads", "time_seconds",
        "time_ms", "speedup", "efficiency", "correct"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in combined:
        writer.writerow({
            **row,
            "speedup": f'{row["speedup"]:.6f}',
            "efficiency": (
                f'{row["efficiency"]:.6f}'
                if row["efficiency"] is not None else "N/A"
            ),
        })

labels = ["1M", "10M", "50M"]
x = np.arange(len(sizes))
width = 0.19

def values(version, threads):
    return [
        get_measurement(version, n, threads)["time"] * 1000
        for n in sizes
    ]

# Graph 1: Execution time, like slide 7 of the reference PPT.
fig, ax = plt.subplots(figsize=(11, 6))
series = [
    ("sequential", 1, "Serial CPU"),
    ("openmp", 4, "OpenMP (4 threads)"),
    ("cuda_kernel", 256, "CUDA kernel only"),
    ("cuda_total", 256, "CUDA total (with copies)"),
]
for i, (version, threads, label) in enumerate(series):
    vals = values(version, threads)
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in vals],
                 padding=3, fontsize=8, rotation=0)

ax.set_yscale("log")
ax.set_ylabel("Execution time (ms, log scale)")
ax.set_xlabel("Array size (N)")
ax.set_title("Execution Time vs Array Size")
ax.set_xticks(x, labels)
ax.grid(axis="y", which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "execution_time.png"), dpi=200)
plt.close(fig)

# Graph 2: Speedup relative to sequential CPU.
fig, ax = plt.subplots(figsize=(10, 6))
speedup_series = [
    ("openmp", 4, "OpenMP (4 threads)"),
    ("cuda_kernel", 256, "CUDA kernel only"),
    ("cuda_total", 256, "CUDA total (with copies)"),
]
for i, (version, threads, label) in enumerate(speedup_series):
    vals = [
        serial[n] / get_measurement(version, n, threads)["time"]
        for n in sizes
    ]
    bars = ax.bar(x + (i - 1) * 0.24, vals, 0.24, label=label)
    ax.bar_label(bars, labels=[f"{v:.2f}x" for v in vals],
                 padding=3, fontsize=9)

ax.axhline(1.0, linestyle="--", linewidth=1, label="Serial baseline (1x)")
ax.set_yscale("log")
ax.set_ylabel("Speedup (log scale)")
ax.set_xlabel("Array size (N)")
ax.set_title("Speedup vs Sequential CPU")
ax.set_xticks(x, labels)
ax.grid(axis="y", which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "speedup.png"), dpi=200)
plt.close(fig)

# Graph 3: OpenMP efficiency only.
fig, ax = plt.subplots(figsize=(10, 6))
for i, threads in enumerate(thread_counts):
    vals = []
    for n in sizes:
        t = get_measurement("openmp", n, threads)["time"]
        vals.append((serial[n] / t) / threads)

    bars = ax.bar(x + (i - 1) * 0.24, vals, 0.24,
                  label=f"{threads} thread{'s' if threads > 1 else ''}")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals],
                 padding=3, fontsize=9)

ax.axhline(1.0, linestyle="--", linewidth=1, label="Ideal efficiency (1.0)")
ax.set_ylabel("Efficiency (speedup / threads)")
ax.set_xlabel("Array size (N)")
ax.set_title("OpenMP Efficiency")
ax.set_xticks(x, labels)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(GRAPHS, "efficiency.png"), dpi=200)
plt.close(fig)

# Report correctness before trusting the plots.
incorrect = [r for r in combined if r["correct"] != "YES"]
if incorrect:
    print("WARNING: Some results failed correctness checks!")
    for r in incorrect:
        print(r)
else:
    print("All measurements report correct=YES.")

print(f"Combined CSV: {combined_path}")
print("Graphs generated:")
print(os.path.join(GRAPHS, "execution_time.png"))
print(os.path.join(GRAPHS, "speedup.png"))
print(os.path.join(GRAPHS, "efficiency.png"))
print("All execution times in graphs are in milliseconds.")
