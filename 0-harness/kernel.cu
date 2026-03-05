#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <thread>


// -----------------------
// Utilities
// -----------------------
#define CUDA_CHECK(call) do { \
    /* TODO (Section 3): implement error checking */ \
    cudaError_t result = call; \
    if(result != cudaSuccess) { \
        fprintf(stderr, \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__, \
                __LINE__, \
                result, \
                cudaGetErrorString(result)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }


// -----------------------
// Section 1: CPU GEMM + stats
// -----------------------
static void fill_random(std::vector<float>& x, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

     // TODO (Section 1): generate random floats and fill in `x` with dist(rng)
    for (int i=0; i<x.size(); i++) {
        x[i] = dist(rng);
    }
   
}

/* A is MxK, B is KxN, C is MxN
 * A(i,k) lives at A[i*K + k]
 * B(k,j) lives at B[k*N + j]
 * C(i,j) lives at C[i*N + j]
*/
static void gemm_cpu_naive(const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int M, int N, int K) {
    // TODO (Section 1): naive gemm with triple loop i, j, k (row-major)
    for (int i=0; i<M; i++) {
        for (int k=0; k<K; k++) {
            for (int j=0; j<N; j++) {
                C[i*N + j] += A[i*K +k] * B[k*N + j];
            }
        }
    }
}

static void print_stats(const char* name, const std::vector<float>& x) {
    double sum = 0.0;
    float max_abs = 0.0f;

    // TODO (Section 1): calculate the sum
    for (int i=0; i<x.size(); i++) {
        sum += x[i];
        max_abs = std::max(max_abs, std::abs(x[i]));
    }

    double avg = x.empty() ? 0.0 : sum / x.size();
    std::printf("%s: avg=%f, max_abs=%f\n", name, avg, max_abs);
}


// -----------------------
// Section 2: naive GPU GEMM kernel
// -----------------------
// One thread computes one output element C(i,j).
__global__ void gemm_gpu_naive(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    // TODO (Section 2): implement naive GEMM kernel

    // We know A points to M*K floats, B points to K*N floats, C points to M*N floats
    // So A(i,k) lives at A[i*K + k]
    // B(k,j) lives at B[k*N + j]
    // C(i,j) lives at C[i*N + j] like before in the CPU GEMM case
    float val = 0.0f;
    for (int k=0; k<K; k++) {
        val += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = val;
}


// -----------------------
// Section 3: harness + correctness helpers
// -----------------------
static float max_abs_error(const std::vector<float>& ref,
                           const std::vector<float>& got) {
    float error = 0.0f;
    for (int i=0; i<ref.size(); i++) {
        error = std::max(error, std::abs(ref[i] - got[i]));
    }
    // TODO (Section 3): compute max |ref - got|
    return error;
}


int main() {
    int M = 256, N = 256, K = 256;

    // TODO (Section 1): allocate host A, B, C, C_ref (row-major)
    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N);
    std::vector<float> h_C_ref(M*N);

    // TODO (Section 1): fill A,B with deterministic random
    fill_random(h_A, 42);
    fill_random(h_B, 67);

    // TODO (Section 1): run CPU GEMM into Cref, print stats
    gemm_cpu_naive(h_A, h_B, h_C_ref, M,N,K);
    print_stats("h_C_ref", h_C_ref);

    // TODO (Section 3): allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));

    // TODO (Section 3): initialize device memory
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));
    

    // TODO (Section 3): launch kernel
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    // launch gemm_gpu_naive
    gemm_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO (Section 5): benchmark and report ms + TFLOPs
    int warmup_iters  = 500;
    int profile_iters = 100;
    cudaEvent_t start, stop;
    float ms;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup_iters; i++) {
        gemm_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord( start, 0 ));
    
    for (int i = 0; i < profile_iters; i++) {
        gemm_gpu_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord( stop, 0 ));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime( &ms, start, stop ));

    float avg_ms = ms / profile_iters;
    float seconds = avg_ms * 1e-3;
    float tflops = (2.0 * M * N * K) / seconds / 1e12;
    std::printf("avg_ms: %f, tflops: %f\n", avg_ms, tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // TODO (Section 3): copy C back to host, print out max absolute error
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("max_abs_error: %f\n", max_abs_error(h_C_ref, h_C));

    // TODO (Section 3): free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
