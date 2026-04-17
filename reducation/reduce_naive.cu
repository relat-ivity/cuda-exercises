#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

#define CHECK_CUDA(call) \
do { \
    cudaError_t err=(call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void reduceUnrolling8(float *g_idata, float *g_odata, int n) {
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    float *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) 
        return;
    for (int i = 1; i < blockDim.x; i *= 2) {
        int index = 2 * i * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + i]; 
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}


extern "C" void solve(const float* input, float* output, int N) {
    int block_size = BLOCK_SIZE;
    int grid_size = (N - 1) / block_size + 1;
    dim3 block(block_size, 1);
    dim3 grid(grid_size, 1);

    float *idata_dev = NULL;
    float *odata_dev = NULL;
    CHECK_CUDA(cudaMalloc(&idata_dev, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&odata_dev, grid_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(idata_dev, input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    reduceUnrolling8<<<grid.x, block.x>>>(idata_dev, odata_dev, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    *output = 0;
    float *result = (float*)malloc(grid_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(result, odata_dev, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < grid_size; i++) 
        *output += result[i];
    
    CHECK_CUDA(cudaFree(idata_dev));
    CHECK_CUDA(cudaFree(odata_dev));
    free(result);
    cout<<output<<endl;
}