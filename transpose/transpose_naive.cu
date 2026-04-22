#include <cuda_runtime.h>

#define UNROLLING_X 4

__global__ void matrix_transpose_kernel(const float* input, float* output, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) output[ix + iy * nx] = input[iy + ix * ny];
}

__global__ void matrix_transpose_kernel_unroll(const float* input, float* output, int nx, int ny) {
    int ix = UNROLLING_X * blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    for (int stride = 0; stride < UNROLLING_X; ++stride) {
        int index_in = iy + (ix + stride * blockDim.x) * ny;
        int index_out = ix + stride * blockDim.x + iy * nx;
        if (ix + stride * blockDim.x < nx && iy < ny) output[index_out] = input[index_in];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows - 1) / threadsPerBlock.x + 1,
                       (cols - 1) / threadsPerBlock.y + 1);
    dim3 blocksPerGridUnroll((rows - 1) / (UNROLLING_X * threadsPerBlock.x) + 1,
                       (cols - 1) / threadsPerBlock.y + 1);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
