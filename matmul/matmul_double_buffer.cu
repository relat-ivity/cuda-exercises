#include<cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (*reinterpret_cast<float4*>(&(pointer)))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_float4(float* A, float* B, float* C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;
    constexpr int thread_num = (BM / TM) * (BN / TN);

    A += by * BM * K;
    B += bx * BN; 

    __shared__ float as[2][BK][BM];
    __shared__ float bs[2][BK][BN];

    float reg_a[2][TM];
    float reg_b[2][TN];
    float reg_acc[TM][TN] = {0};

    constexpr int asLoadPerThread = BM * BK / (4 * thread_num);
    constexpr int bsLoadPerThread = BK * BN / (4 * thread_num);
    float reg_load_a[4 * asLoadPerThread];
    float reg_load_b[4 * bsLoadPerThread];

    constexpr int as_col_num = BK / 4;
    int as_col = 4 * (tid % as_col_num);
    int as_row_base = tid / as_col_num;
    constexpr int as_row_stride = thread_num / as_col_num;

    constexpr int bs_col_num = BN / 4;
    int bs_col = 4 * (tid % bs_col_num); 
    int bs_row_base = tid / bs_col_num;
    constexpr int bs_row_stride = thread_num / bs_col_num;

    // load as
    #pragma unroll
    for(int l = 0; l < asLoadPerThread; l++) {
        int as_row = as_row_base + l * as_row_stride;
        FETCH_FLOAT4(reg_load_a[l * 4]) = FETCH_FLOAT4(A[as_row * K + as_col]);
        as[0][as_col][as_row] = reg_load_a[l * 4];
        as[0][as_col + 1][as_row] = reg_load_a[l * 4 + 1];
        as[0][as_col + 2][as_row] = reg_load_a[l * 4 + 2];
        as[0][as_col + 3][as_row] = reg_load_a[l * 4 + 3];
    }

    // load bs
    #pragma unroll
    for(int l = 0; l < bsLoadPerThread; l++) {
        int bs_row = bs_row_base + l * bs_row_stride;
        FETCH_FLOAT4(bs[0][bs_row][bs_col]) = FETCH_FLOAT4(B[bs_row * N + bs_col]);
    }

    __syncthreads();

    #pragma unroll
    for(int l = 0; l < TM; l += 4) { 
        FETCH_FLOAT4(reg_a[0][l]) = FETCH_FLOAT4(as[0][0][ty * TM + l]);
    }

    #pragma unroll
    for(int l = 0; l < TN; l += 4) {
        FETCH_FLOAT4(reg_b[0][l]) = FETCH_FLOAT4(bs[0][0][tx * TN + l]);
    }

    int k_base = 0;
    int read_buffer_id = 0;
    int write_buffer_id = 1;
    while (k_base < K) {
        k_base += BK;
        if(k_base < K) {
            #pragma unroll
            for(int l = 0; l < asLoadPerThread; l++) {
                int as_row = as_row_base + l * as_row_stride;
                FETCH_FLOAT4(reg_load_a[l * 4]) = FETCH_FLOAT4(A[as_row * K + as_col + k_base]);
            }

            #pragma unroll
            for(int l = 0; l < bsLoadPerThread; l++) {
                int bs_row = bs_row_base + l * bs_row_stride;
                FETCH_FLOAT4(reg_load_b[l * 4]) = FETCH_FLOAT4(B[(bs_row + k_base) * N + bs_col]);
            }   
        }

        #pragma unroll
        for (int k = 0; k < BK - 1; k++) {
            #pragma unroll
            for(int l = 0; l < TM; l += 4) {
                FETCH_FLOAT4(reg_a[(k + 1) % 2][l]) = FETCH_FLOAT4(as[read_buffer_id][k + 1][ty * TM + l]);
            }
            
            #pragma unroll
            for (int l = 0; l < TN; l +=4) {
                FETCH_FLOAT4(reg_b[(k + 1) % 2][l]) = FETCH_FLOAT4(bs[read_buffer_id][k + 1][tx * TN + l]);
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    reg_acc[i][j] += reg_a[k % 2][i] * reg_b[k % 2][j];
                }
            }
        }

        if (k_base < K) {
            #pragma unroll
            for (int l = 0; l < asLoadPerThread; l++) {
                int as_row = as_row_base + l * as_row_stride;
                as[write_buffer_id][as_col][as_row] = reg_load_a[l * 4];
                as[write_buffer_id][as_col + 1][as_row] = reg_load_a[l * 4 + 1];
                as[write_buffer_id][as_col + 2][as_row] = reg_load_a[l * 4 + 2];
                as[write_buffer_id][as_col + 3][as_row] = reg_load_a[l * 4 + 3];
            }

            #pragma unroll
            for (int l = 0; l < bsLoadPerThread; l++) { 
                int bs_row = bs_row_base + l * bs_row_stride;
                FETCH_FLOAT4(bs[write_buffer_id][bs_row][bs_col]) = FETCH_FLOAT4(reg_load_b[l * 4]);
            }

            __syncthreads();

            #pragma unroll
            for (int l = 0; l < TM; l += 4) {
                FETCH_FLOAT4(reg_a[0][l]) = FETCH_FLOAT4(as[write_buffer_id][0][TM * ty + l]);
            }

            #pragma unroll
            for (int l = 0; l < TN; l += 4) {
                FETCH_FLOAT4(reg_b[0][l]) = FETCH_FLOAT4(bs[write_buffer_id][0][TN * tx + l]);
            }
        }

        #pragma unroll
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                reg_acc[i][j] += reg_a[1][i] * reg_b[1][j];
            }
        }

        read_buffer_id ^= 1;
        write_buffer_id ^= 1;
    }

    int c_row_base = by * BM + ty * TM; 
    int c_col_base = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int c_row = c_row_base + i;
            int c_col = c_col_base + j;
            FETCH_FLOAT4(C[c_row * N + c_col]) = FETCH_FLOAT4(reg_acc[i][j]);
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
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
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            if((C_row + i) < M && (C_col + j) < N) {
                C[(C_row + i) * N + C_col + j] = reg_c[i][j];
            }
        }
    }
}

extern "C" void solve(float* A, float* B, float* C, int M, int K, int N) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N - 1) / BN + 1, (M - 1) / BM + 1);
    if (K % 4 == 0 && N % 4 == 0 && M % BM == 0 && N % BN == 0 && K % BK == 0 ) {
        matmul_float4<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, K, N);
    } else {
        matmul<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, K, N);
    }
    cudaDeviceSynchronize();
}