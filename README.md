# CUDA Matrix Multiplication Benchmark

A small CUDA/C++ project comparing three matrix-multiplication implementations:

1. CPU baseline
2. Naive CUDA kernel
3. Tiled CUDA kernel using shared memory

The goal is to explore GPU execution, memory access patterns, CUDA block/grid configuration, and basic performance profiling.

## Why this project

This project was built as a hands-on CUDA performance exercise. It focuses on topics that matter in GPU programming:

- CUDA kernel design
- host/device memory management
- grid and block configuration
- shared-memory tiling
- synchronization with `__syncthreads()`
- GPU timing with CUDA events
- correctness checking against a CPU implementation

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- CMake 3.18+ (optional)

## Build

### With nvcc

```bash
nvcc -O3 -std=c++17 src/main.cu -o cuda_matmul
```

### With CMake

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Run

```bash
./cuda_matmul
```

Optionally pass matrix size:

```bash
./cuda_matmul 1024
```

The program prints execution times for the CPU, naive CUDA, and tiled CUDA implementations and verifies GPU results against the CPU baseline.

## Implementation notes

### Naive CUDA kernel

Each GPU thread computes one output matrix element directly from global memory.

### Tiled CUDA kernel

The optimized kernel divides the matrices into tiles and stages input data in CUDA shared memory. Threads within a block reuse the cached tile data before loading the next tile.

This reduces repeated global-memory accesses and demonstrates a common CUDA optimization pattern.

## Suggested profiling extensions

This repository is intentionally small enough to extend. Useful next steps include:

- Compare tile sizes such as 8x8, 16x16, and 32x32
- Measure host-to-device and device-to-host transfer times separately
- Profile with NVIDIA Nsight Compute
- Investigate memory coalescing
- Add pinned host memory
- Compare against cuBLAS SGEMM
- Calculate effective GFLOP/s

## Benchmark results

Hardware-dependent benchmark numbers are intentionally not committed. Run the project on your CUDA-capable GPU and add your own results here.

| Matrix Size | CPU | Naive CUDA | Tiled CUDA |
|---|---:|---:|---:|
| 512 x 512 | TBD | TBD | TBD |
| 1024 x 1024 | TBD | TBD | TBD |

## Project structure

```text
cuda-matrix-multiplication-benchmark/
├── CMakeLists.txt
├── README.md
├── .gitignore
└── src/
    └── main.cu
```

## License

MIT
