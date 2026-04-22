#include <cuda_runtime.h>

// 合并存储的基础上，一个线程处理多个转置
template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void matrix_transpose_kernel_tile(const float* input, float* output, int M, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ix = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int iy = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // 按行写入shared memory
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1]; // padding解决bank conflict
    if(ix < N) {
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += TILE_SIZE) {
            if (iy + i < M) sdata[ty + i][tx] = input[(iy + i) * N + ix];
        }
    } 
    __syncthreads();

    // block内交换xy，实现按行写入
    ix = blockIdx.y * BLOCK_SIZE + tx;
    iy = blockIdx.x * BLOCK_SIZE + ty;
    if (ix < M) {
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE; i += TILE_SIZE) {
            if (iy + i < N) output[(iy + i) * M + ix] = sdata[tx][ty + i];
        }
    } 
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int TILE_SIZE = 8;

    dim3 threadsPerBlock(BLOCK_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N - 1) / BLOCK_SIZE + 1,
                       (M - 1) / BLOCK_SIZE + 1);

    matrix_transpose_kernel_tile<BLOCK_SIZE, TILE_SIZE><<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
}
