// shared memory 分块 + 向量化访存
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (*reinterpret_cast<float4*>(&(pointer)))

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrix_multiplication_kernel_float4(float* A, float* B, float* C, int M, int K, int N) {
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    // 偏移到块首地址
    A += by * BM * K;
    B += bx * BN;

    __shared__ float as[BM][BK];
    __shared__ float bs[BK][BN];
    
    // 向量化存储
    unsigned int as_row_thread_num = BK / 4;
    unsigned int bs_row_thread_num = BN / 4;

    unsigned int tid = ty * blockDim.x + tx;

    unsigned int as_row_idx = tid / as_row_thread_num;
    unsigned int as_col_idx = (tid % as_row_thread_num) * 4;
    unsigned int bs_row_idx = tid / bs_row_thread_num;
    unsigned int bs_col_idx = (tid % bs_row_thread_num) * 4;

    unsigned int a_block_stride = blockDim.x * blockDim.y / as_row_thread_num;
    unsigned int b_block_stride = blockDim.x * blockDim.y / bs_row_thread_num;

    float reg_b[TN];
    float reg_a;
    float reg_c[TM][TN] = {0};

    // BK循环
    for (int k_tile = 0; k_tile < K ; k_tile += BK) {
        // GMEM -> SMEM
        #pragma unroll
        for (int stride = 0; stride < BM; stride += a_block_stride) {
            FETCH_FLOAT4(as[as_row_idx + stride][as_col_idx]) = 
                    FETCH_FLOAT4(A[(as_row_idx + stride) * K + as_col_idx + k_tile]);
        }

        #pragma unroll
        for (int stride = 0; stride < BK; stride += b_block_stride) {
            FETCH_FLOAT4(bs[bs_row_idx + stride][bs_col_idx]) = 
                    FETCH_FLOAT4(B[(bs_row_idx + stride + k_tile) * N + bs_col_idx]);
        }

        __syncthreads();

        // BK内循环
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            
            // 读取寄存器
            #pragma unroll
            for (int j = 0; j < TN; j+=4) {
                FETCH_FLOAT4(reg_b[j]) = FETCH_FLOAT4(bs[k][tx * TN + j]); 
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a = as[ty * TM + i][k];
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a * reg_b[j];
                }
            } 
        }
        __syncthreads();
    }

    unsigned int C_row = by * BM + ty * TM;
    unsigned int C_col = bx * BN + tx * TN;
    #pragma unroll
    for (int i = 0; i < TM; i++){
        for (int j = 0; j < TN; j += 4) {
            FETCH_FLOAT4(C[(C_row + i) * N + C_col + j]) = FETCH_FLOAT4(reg_c[i][j]);
        } 
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrix_multiplication_kernel(float* A, float* B, float* C, int M, int K, int N) {
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    __shared__ float as[BM][BK];
    __shared__ float bs[BK][BN];
    
    unsigned int tid = ty * blockDim.x + tx;

    unsigned int as_row_idx = tid / BK;
    unsigned int as_col_idx = tid % BK;
    unsigned int bs_row_idx = tid / BN;
    unsigned int bs_col_idx = tid % BN;

    unsigned int a_block_stride = blockDim.x * blockDim.y / BK;
    unsigned int b_block_stride = blockDim.x * blockDim.y / BN;

    float reg_b[TN];
    float reg_a;
    float reg_c[TM][TN] = {0};

    // BK循环
    for (int k_tile = 0; k_tile < K ; k_tile += BK) {
        // GMEM -> SMEM
        #pragma unroll
        for (int stride = 0; stride < BM; stride += a_block_stride) {
            if ((by * BM + as_row_idx + stride) < M && (as_col_idx + k_tile) < K) {
                as[as_row_idx + stride][as_col_idx] = A[(by * BM + as_row_idx + stride) * K + as_col_idx + k_tile];
            } else {
                as[as_row_idx + stride][as_col_idx] = 0.0f;
            }
        }

        #pragma unroll
        for (int stride = 0; stride < BK; stride += b_block_stride) {
            if ((bs_row_idx + stride + k_tile) < K && (bx * BN + bs_col_idx) < N) {
                bs[bs_row_idx + stride][bs_col_idx] = B[(bs_row_idx + stride + k_tile) * N + bx * BN + bs_col_idx];
            } else {
                bs[bs_row_idx + stride][bs_col_idx] = 0.0f;
            }
        }

        __syncthreads();

        // BK内循环
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 读取寄存器
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = bs[k][tx * TN + j]; 
            }

            // 计算
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a = as[ty * TM + i][k];
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a * reg_b[j];
                }
            } 
        }
        __syncthreads();
    }

    unsigned int C_row = by * BM + ty * TM;
    unsigned int C_col = bx * BN + tx * TN;
    #pragma unroll
    for (int i = 0; i < TM; i++){
        for (int j = 0; j < TN; j++) {
            if((C_row + i) < M && (C_col + j) < N) {
                C[(C_row + i) * N + C_col + j] = reg_c[i][j];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(float* A, float* B, float* C, int M, int K, int N) {
    dim3 threadsPerBlock(BM / TM, BN / TN);
    dim3 blocksPerGrid((N - 1) / BN + 1,
                       (M - 1) / BM + 1);

    if (M % BM == 0 && N % BN == 0 && K % BK == 0 && N % 4 == 0 && K % 4 ==0) {
        matrix_multiplication_kernel_float4<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    } else {
        matrix_multiplication_kernel<BM, BN, BK, TM, TN><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    }
    
    cudaDeviceSynchronize();
}
