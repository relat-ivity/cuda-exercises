#include <cuda_runtime.h>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;   // [0, BM/TM)
    int ty = threadIdx.y;   // [0, BN/TN)

    int row = bx * BM + tx * TM; // 行起点
    int col = by * BN + ty * TN; // 列起点

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 每个线程的累加寄存器
    float acc[TM][TN] = {0};

    for (int kb = 0; kb < K; kb += BK) {

        /* -------- load A tile -------- */
        for (int i = tx; i < BM; i += blockDim.x) {
            for (int k = ty; k < BK; k += blockDim.y) {
                int ai = bx * BM + i;
                int ak = kb + k;
                if (ai < M && ak < K)
                    As[i][k] = A[ai * K + ak];
                else
                    As[i][k] = 0.0f;
            }
        }

        /* -------- load B tile -------- */
        for (int k = tx; k < BK; k += blockDim.x) {
            for (int j = ty; j < BN; j += blockDim.y) {
                int bk2 = kb + k;
                int bj = by * BN + j;
                if (bk2 < K && bj < N)
                    Bs[k][j] = B[bk2 * N + bj];
                else
                    Bs[k][j] = 0.0f;
            }
        }

        __syncthreads();

        /* -------- compute -------- */
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    acc[i][j] +=
                        As[tx * TM + i][k] *
                        Bs[k][ty * TN + j];
                }
            }
        }

        __syncthreads();
    }

    /* -------- write back -------- */
    for (int i = 0; i < TM; i++) {
        int gi = row + i;
        if (gi < M) {
            for (int j = 0; j < TN; j++) {
                int gj = col + j;
                if (gj < N) {
                    C[gi * N + gj] = acc[i][j];
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 threadsPerBlock(BM / TM, BN / TN);
    dim3 blocksPerGrid((M + BM - 1) / BM,
                       (N + BN - 1) / BN);

    matrix_multiplication_kernel<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
