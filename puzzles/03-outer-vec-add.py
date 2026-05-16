"""
Puzzle 03: Outer Vector Add
==============
In this puzzle we will enter the 2D world!

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Consider an outer vector addition operation. The result is a matrix where
each element (i, j) is the sum of A[i] and B[j].

The main difference from the previous puzzle is that C is now a 2D tensor and
we have two different iterators in buffers A and B. So the dataflow is also
a little different.

But remeMber that any N dimensional tensor can be viewed as a 1D tensor in memory.
So we just need to handle the indexing properly.

03-1: Outer vector addition.

Inputs:
    A: Tensor([N,], float16)  # input tensor
    B: Tensor([M,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 8192
    M: int   # size of the tensor. 1 <= M <= 8192

Output:
    C: [N, M]  # output tensor

Definition:
    for i in range(N):
        for j in range(M):
            C[i, j] = A[i] + B[j]
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor):
    assert len(A.shape) == 1
    assert len(B.shape) == 1
    assert A.dtype == B.dtype == torch.float16
    return torch.add(input=A[:, None], other=B[None, :])


def print_cuda_source(puzzle_tl, tl_hyper_params: dict, name: str):
    print(f"\n=== Generated CUDA: {name} ===\n")
    tl_kernel = puzzle_tl.compile(**tl_hyper_params)
    print(tl_kernel.get_kernel_source())


@tilelang.jit
def tl_outer_add_naive(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (bx, by):
        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
            C[bx * BLOCK_N + i, by * BLOCK_M + j] = A[bx * BLOCK_N + i] + B[by * BLOCK_M + j]

    return C


@tilelang.jit
def tl_outer_add_v2(A, B, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float16
    A: T.Tensor((N,), dtype)
    B: T.Tensor((M,), dtype)
    C = T.empty((N, M), dtype)

    # TODO: Implement this function
    with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=256) as (bx, by):
        tmp_A = T.alloc_fragment((BLOCK_N, ), dtype)
        tmp_B = T.alloc_fragment((BLOCK_M, ), dtype)
        tmp_C = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)

        T.copy(A[bx * BLOCK_N], tmp_A)
        T.copy(B[by * BLOCK_M], tmp_B)

        for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                tmp_C[i, j] = tmp_A[i] + tmp_B[j]

        T.copy(tmp_C, C[bx * BLOCK_N, by * BLOCK_M])
    return C

def run_outer_add_benchmark():
    print("\n=== Outer Vector Add Benchmark ===\n")
    N = 8192 * 4
    M = 4096 * 4
    BLOCK_N = 64
    BLOCK_M = 64
    tl_hyper_params = {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M}

    print_cuda_source(tl_outer_add_naive, tl_hyper_params, "tl_outer_add_naive")
    print_cuda_source(tl_outer_add_v2, tl_hyper_params, "tl_outer_add_v2")

    bench_puzzle(
        tl_outer_add_naive,
        ref_outer_add,
        tl_hyper_params,
        bench_name="TileLang outer add",
        bench_torch=True,
    )


    bench_puzzle(
        tl_outer_add_v2,
        ref_outer_add,
        tl_hyper_params,
        bench_name="TileLang outer add",
        bench_torch=True,
    )

def run_outer_add():
    print("\n=== Outer Vector Add ===\n")
    N = 8192
    M = 4096
    BLOCK_N = 1024
    BLOCK_M = 1024
    test_puzzle(
        tl_outer_add_naive,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )

    test_puzzle(
        tl_outer_add_v2,
        ref_outer_add,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )


def run_outer_add_workload_sweep(out_path: str = "outer_add_sweep.png"):
    """Sweep workloads and plot torch / naive / v2 kernel time + bandwidth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLOCK_N = 64
    BLOCK_M = 64

    # Square-ish total-size sweep.
    size_sweep = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 8192),
        (32768, 16384),
    ]

    # Aspect-ratio sweep at fixed total size = 16M elements.
    shape_sweep = [
        (64, 262144),
        (256, 65536),
        (1024, 16384),
        (4096, 4096),
        (16384, 1024),
        (65536, 256),
        (262144, 64),
    ]

    kernels = [
        ("naive", tl_outer_add_naive),
        ("v2", tl_outer_add_v2),
    ]

    def bench_one(N, M):
        params = {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M}
        row = {"N": N, "M": M, "elems": N * M}
        first = True
        for name, fn in kernels:
            res = bench_puzzle(
                fn,
                ref_outer_add,
                params,
                bench_name=name,
                bench_torch=first,
                verbose=False,
            )
            row[name] = res["tl"]
            if first:
                row["torch"] = res["torch"]
                first = False
        # fp16 = 2 bytes; read N+M, write N*M.
        bytes_moved = (N * M + N + M) * 2
        for k in ("torch", "naive", "v2"):
            row[f"{k}_bw"] = bytes_moved / (row[k] * 1e-3) / 1e9  # GB/s
        return row

    def run_sweep(sweep, title):
        print(f"\n=== Sweep: {title} ===")
        rows = []
        for N, M in sweep:
            print(f"  Benchmarking N={N}, M={M} ...")
            rows.append(bench_one(N, M))
        header = (
            f"{'N':>7} {'M':>7} {'elems':>10}  "
            f"{'torch ms':>9} {'naive ms':>9} {'v2 ms':>9}  "
            f"{'torch GB/s':>10} {'naive GB/s':>10} {'v2 GB/s':>10}  {'v2/torch':>8}"
        )
        print(header)
        for r in rows:
            print(
                f"{r['N']:>7} {r['M']:>7} {r['elems']:>10}  "
                f"{r['torch']:>9.3f} {r['naive']:>9.3f} {r['v2']:>9.3f}  "
                f"{r['torch_bw']:>10.1f} {r['naive_bw']:>10.1f} {r['v2_bw']:>10.1f}  "
                f"{r['v2'] / r['torch']:>8.2f}"
            )
        return rows

    size_rows = run_sweep(size_sweep, "square size sweep")
    shape_rows = run_sweep(shape_sweep, "aspect-ratio sweep (16M elements)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def plot_panel(ax, rows, x_key, x_label, y_keys, y_label, title, log_x=True):
        xs = [r[x_key] for r in rows]
        for k, marker, color in y_keys:
            ys = [r[k] for r in rows]
            ax.plot(xs, ys, marker=marker, color=color, label=k)
        if log_x:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    time_keys = [("torch", "o", "#444"), ("naive", "s", "tab:orange"), ("v2", "^", "tab:blue")]
    bw_keys = [("torch_bw", "o", "#444"), ("naive_bw", "s", "tab:orange"), ("v2_bw", "^", "tab:blue")]

    plot_panel(
        axes[0, 0], size_rows, "elems", "N*M (output elements)", time_keys, "time (ms)",
        "Square sweep: kernel time",
    )
    plot_panel(
        axes[0, 1], size_rows, "elems", "N*M (output elements)", bw_keys, "effective BW (GB/s)",
        "Square sweep: bandwidth",
    )

    for r in shape_rows:
        r["shape_label"] = f"{r['N']}x{r['M']}"
    plot_panel(
        axes[1, 0], shape_rows, "N", "N (M = 16M/N)", time_keys, "time (ms)",
        "Shape sweep @ 16M elems: kernel time",
    )
    plot_panel(
        axes[1, 1], shape_rows, "N", "N (M = 16M/N)", bw_keys, "effective BW (GB/s)",
        "Shape sweep @ 16M elems: bandwidth",
    )
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xticks([r["N"] for r in shape_rows])
        ax.set_xticklabels([r["shape_label"] for r in shape_rows], rotation=30, ha="right")

    fig.suptitle(f"Outer vector add: kernel speed across workloads (BLOCK={BLOCK_N}x{BLOCK_M})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure to: {out_path}")

    _plot_roofline(size_rows, shape_rows, out_path="outer_add_roofline.png")


def _plot_roofline(size_rows, shape_rows, out_path: str):
    """Plot an H100 roofline with our measured (AI, achieved FLOPS) points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # H100 SXM5 ceilings (FP16, no tensor cores — element-wise add can't use them).
    HBM_BW = 3000.0  # GB/s
    FP16_PEAK = 67000.0  # GFLOPS (CUDA cores)
    ridge_AI = FP16_PEAK / HBM_BW  # FLOP/byte where lines cross

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Roofline.
    ai_grid = np.logspace(-2, 2, 400)
    bw_line = HBM_BW * ai_grid
    roof = np.minimum(bw_line, FP16_PEAK)
    ax.plot(ai_grid, roof, color="black", linewidth=2, label="H100 roofline")
    ax.axvline(ridge_AI, color="gray", linestyle=":", alpha=0.6)
    ax.text(ridge_AI * 1.15, FP16_PEAK * 0.55, f"ridge AI ≈ {ridge_AI:.0f}", color="gray")
    # Annotate ceilings.
    ax.text(0.012, FP16_PEAK * 1.1, f"FP16 peak (CUDA cores) ≈ {FP16_PEAK / 1000:.0f} TFLOPS",
            color="black", fontsize=9)
    ax.text(0.013, HBM_BW * 0.013 * 1.4, f"HBM BW ≈ {HBM_BW / 1000:.1f} TB/s",
            color="black", fontsize=9, rotation=33)

    def points_for(rows, label_prefix):
        # AI = N*M / (2*(N*M + N + M)) ≈ 0.5
        out = {"torch": [], "naive": [], "v2": []}
        for r in rows:
            ai = r["elems"] / (2.0 * (r["elems"] + r["N"] + r["M"]))
            for k in ("torch", "naive", "v2"):
                # achieved GFLOPS = (N*M ops) / time_ms / 1e6
                gflops = r["elems"] / (r[k] * 1e-3) / 1e9
                out[k].append((ai, gflops, f"{label_prefix} {r['N']}x{r['M']}"))
        return out

    size_pts = points_for(size_rows, "sq")
    shape_pts = points_for(shape_rows, "sh")

    style = {
        "torch": ("o", "#444"),
        "naive": ("s", "tab:orange"),
        "v2": ("^", "tab:blue"),
    }
    for kernel in ("torch", "naive", "v2"):
        marker, color = style[kernel]
        all_pts = size_pts[kernel] + shape_pts[kernel]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        ax.scatter(xs, ys, marker=marker, color=color, s=55, alpha=0.85,
                   edgecolors="black", linewidths=0.4, label=kernel, zorder=5)

    # Annotate the largest-size point per kernel.
    largest = size_rows[-1]
    ai_l = largest["elems"] / (2.0 * (largest["elems"] + largest["N"] + largest["M"]))
    for kernel in ("torch", "naive", "v2"):
        gflops = largest["elems"] / (largest[kernel] * 1e-3) / 1e9
        ax.annotate(
            f"{kernel} @ {largest['N']}x{largest['M']}\n  {gflops/1000:.2f} TFLOPS",
            xy=(ai_l, gflops), xytext=(ai_l * 1.6, gflops * (0.45 if kernel == "torch" else 0.7)),
            fontsize=8, color=style[kernel][1],
            arrowprops=dict(arrowstyle="->", color=style[kernel][1], lw=0.8),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e1, 2e5)
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved performance (GFLOPS)")
    ax.set_title("Outer vector add — H100 roofline\n"
                 "(op AI ≈ 0.5, far below ridge ≈ 22 → memory-bound)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved roofline to: {out_path}")


if __name__ == "__main__":
    run_outer_add()
    run_outer_add_workload_sweep()
