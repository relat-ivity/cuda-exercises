#include <cuda_runtime.h>

// block内按行访存，按行写入
template<int BLOCK_SIZE>
__global__ void matrix_transpose_kernel(const float* input, float* output, int M, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // 按行写入shared memory
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1]; // padding解决bank conflict
    if(ix < N && iy < M) sdata[ty][tx] = input[iy * N + ix];
    __syncthreads();

    // block内交换xy，实现按行写入
    ix = blockIdx.y * blockDim.y + tx;
    iy = blockIdx.x * blockDim.x + ty;
    if (ix < M && iy < N) output[iy * M + ix] = sdata[tx][ty];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    constexpr int BLOCK_SIZE = 16;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N - 1) / threadsPerBlock.x + 1,
                       (M - 1) / threadsPerBlock.y + 1);

    matrix_transpose_kernel<BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
