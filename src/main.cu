#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                           \
        if (error__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(error__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int TILE_SIZE = 16;

__global__ void matmul_naive(const float* A,
                             const float* B,
                             float* C,
                             int n)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

__global__ void matmul_tiled(const float* A,
                             const float* B,
                             float* C,
                             int n)
{
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    const int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * TILE_SIZE + threadIdx.x;
        const int b_row = tile * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;

        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

void matmul_cpu(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int n)
{
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

bool verify(const std::vector<float>& reference,
            const std::vector<float>& result,
            float tolerance = 1e-2f)
{
    if (reference.size() != result.size()) {
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float diff = std::fabs(reference[i] - result[i]);
        const float scale = std::max(1.0f, std::fabs(reference[i]));

        if (diff > tolerance * scale) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << reference[i]
                      << ", got " << result[i] << '\n';
            return false;
        }
    }

    return true;
}

template <typename Kernel>
float run_gpu_kernel(Kernel kernel,
                     const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int n,
                     dim3 grid,
                     dim3 block)
{
    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    kernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

int main(int argc, char** argv)
{
    int n = 512;

    if (argc > 1) {
        n = std::stoi(argv[1]);
        if (n <= 0) {
            throw std::invalid_argument("Matrix size must be positive.");
        }
    }

    const std::size_t element_count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = element_count * sizeof(float);

    std::cout << "Matrix size: " << n << " x " << n << "\n";
    std::cout << "Tile size:   " << TILE_SIZE << " x " << TILE_SIZE << "\n\n";

    std::vector<float> h_A(element_count);
    std::vector<float> h_B(element_count);
    std::vector<float> h_C_cpu(element_count, 0.0f);
    std::vector<float> h_C_naive(element_count, 0.0f);
    std::vector<float> h_C_tiled(element_count, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < element_count; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    std::cout << "Running CPU baseline...\n";
    const auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, n);
    const auto cpu_stop = std::chrono::high_resolution_clock::now();

    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((n + block.y - 1) / block.y));

    std::cout << "Running naive CUDA kernel...\n";
    const float naive_ms =
        run_gpu_kernel(matmul_naive, d_A, d_B, d_C, n, grid, block);

    CUDA_CHECK(cudaMemcpy(
        h_C_naive.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Running tiled CUDA kernel...\n";
    const float tiled_ms =
        run_gpu_kernel(matmul_tiled, d_A, d_B, d_C, n, grid, block);

    CUDA_CHECK(cudaMemcpy(
        h_C_tiled.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    const bool naive_ok = verify(h_C_cpu, h_C_naive);
    const bool tiled_ok = verify(h_C_cpu, h_C_tiled);

    std::cout << "\nResults\n";
    std::cout << "-------\n";
    std::cout << "CPU baseline: " << cpu_ms << " ms\n";
    std::cout << "Naive CUDA:   " << naive_ms << " ms"
              << " | verification: " << (naive_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "Tiled CUDA:   " << tiled_ms << " ms"
              << " | verification: " << (tiled_ok ? "PASS" : "FAIL") << "\n";

    if (naive_ms > 0.0f) {
        std::cout << "CPU / naive CUDA speedup: "
                  << cpu_ms / naive_ms << "x\n";
    }

    if (tiled_ms > 0.0f) {
        std::cout << "CPU / tiled CUDA speedup: "
                  << cpu_ms / tiled_ms << "x\n";
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return (naive_ok && tiled_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
