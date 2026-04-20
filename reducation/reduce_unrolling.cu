#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define UNROLLING_SIZE 8

#define CHECK_CUDA(call) \
do { \
    cudaError_t err=(call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void reduceUnrolling8(const float *input, float *output, int n) {
    __shared__ float idata[BLOCK_SIZE]; 
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * UNROLLING_SIZE + threadIdx.x;

    float unrolling_result = 0.0f;
    for(int i = 0; i < UNROLLING_SIZE; i++) {
        int unrolling_index = idx + i * blockDim.x;
        unrolling_result += (unrolling_index < n) ? input[unrolling_index] : 0.0f;
    }
    idata[tid] = unrolling_result;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i /= 2) {
        if (tid < i) {
            idata[tid] += idata[tid + i]; 
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *vdata = idata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    if (tid == 0) { 
        atomicAdd(output, idata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int grid_size = (N - 1) / (UNROLLING_SIZE * BLOCK_SIZE) + 1;
    reduceUnrolling8<<<grid_size, BLOCK_SIZE>>>(input, output, N);
}

int main() {
    const int N = 8;
    float h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_output = 0.0f;

    float *d_input = NULL;
    float *d_output = NULL;

    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    solve(d_input, d_output, N);

    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("result = %f\n", h_output);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    return 0;
}