// nvcc -o test -arch sm_86 test.cu

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <random>

using namespace nvcuda;
using namespace nvcuda::wmma;
#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 8

// GEMM configuration.
#define M_TILES 16
#define N_TILES 16
#define K_TILES 32

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

// --- Based on https://github.com/wzsh/wmma_tensorcore_sample/blob/master/matrix_wmma/matrix_wmma/main.cu ---
__global__ void WMMAF16TensorCore(const float *A, const float *B, float *C)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    // AB = A * B
    int a_col, a_row, b_col, b_row, c_col, c_row;
    a_row = ix * M;
    b_row = iy * N;
    c_col = b_row;
    c_row = a_row;
    // wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);
    wmma::fill_fragment(c_frag, 0.0);
    for (int k = 0; k < K_TOTAL; k += K) {
        a_col = b_col = k;

        if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
            wmma::load_matrix_sync(b_frag, B + b_row + b_col * K_TOTAL, K_TOTAL);            

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    wmma::store_matrix_sync(C + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
}

cudaError_t CalcWMMA(const float *A, const float *B, float *C)
{
    dim3 gridDim, blockDim;
    // 16 warps in one block
    blockDim.x = 4 * WARP_SIZE;
    blockDim.y = 4;

    gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
    gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

    WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C);
    return cudaDeviceSynchronize();
}
// --- End ---

// --- Taken from https://github.com/mlecauchois/micrograd-cuda/blob/main/micrograd_cuda/operations.cu ---
// Matrix multiplication
// a is a_rows x a_cols, b is a_cols x b_cols
__global__ void matmul_kernel(float *a, float *b, float *c, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < a_rows && col < b_cols) {
        float sum = 0.0;
        for(int i = 0; i < a_cols; i++) {
            sum += a[row * a_cols + i] * b[i * b_cols + col];
        }
        c[row * b_cols + col] = sum;
    }
}

extern "C" void matmul_on_gpu(float *d_a, float *d_b, float *d_c, int a_rows, int a_cols, int b_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((b_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (a_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, a_rows, a_cols, b_cols);

    cudaDeviceSynchronize();
}
// ---------------------------------------------------


void fillMat(float* a, size_t rows, size_t cols, size_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0f, 1.0);
    for (size_t i = 0; i < rows*cols; i++) {
        a[i] = dist(rng);
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status) << '\n';
        exit(1);
    }
}

// --- CPU version ---
void cpu_mma(const float* a, const float* b, float* c, size_t mt, size_t nt, size_t kt) {
    for (size_t i = 0; i < mt; i++) {
        for (size_t j = 0; j < nt; j++) {
            float sum = 0.0;
            for (size_t k = 0; k < kt; k++) {
                sum += a[i*kt + k] * b[k*nt + j];
            }
            c[i*nt + j] = sum;
        }
    }
}
// --- End ---

std::pair<double, double> avg_err(float* a, float* b, size_t elems) {
    double err = 0.0;
    double largest = 0.0;
    for (size_t i = 0; i < elems; i++) {
        double lerr = std::abs(double(a[i]) - double(b[i]));
        largest = std::max(largest, lerr);
        err += lerr;
    }
    return {err/double(elems) , largest};
}

int main() {
    float* A, *B;
    float* C1, *C2, *C3;
    cudaMallocManaged((void **)&A, sizeof(float) * M_TOTAL * K_TOTAL);
    cudaMallocManaged((void **)&B, sizeof(float) * K_TOTAL * N_TOTAL);
    cudaMallocManaged((void **)&C1, sizeof(float) * M_TOTAL * N_TOTAL);
    cudaMallocManaged((void **)&C2, sizeof(float) * M_TOTAL * N_TOTAL);
    cudaMallocManaged((void **)&C3, sizeof(float) * M_TOTAL * N_TOTAL);
    
    fillMat(A, M_TOTAL, K_TOTAL, 1);
    fillMat(B, K_TOTAL, N_TOTAL, 2);
    fillMat(C1, M_TOTAL, N_TOTAL, 3);
    fillMat(C2, M_TOTAL, N_TOTAL, 3);
    fillMat(C3, M_TOTAL, N_TOTAL, 3);
    
    cudaCheck(CalcWMMA(A, B, C1));
    matmul_on_gpu(A, B, C2, M_TOTAL, K_TOTAL, N_TOTAL);
    cpu_mma(A, B, C3, M_TOTAL, N_TOTAL, K_TOTAL);

    std::pair<double, double> gpu_cpu_err = avg_err(C2, C3, M_TOTAL * N_TOTAL);
    std::pair<double, double> wmma_cpu_err = avg_err(C1, C3, M_TOTAL * N_TOTAL);
    std::cout << "Old error vs CPU: " << gpu_cpu_err.first << std::endl;
    std::cout << "Tensor error vs CPU: " << wmma_cpu_err.first << std::endl;
    cudaFree(C2);
    cudaFree(C1);
    cudaFree(B);
    cudaFree(A);
}