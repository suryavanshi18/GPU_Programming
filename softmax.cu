

#include <cuda_runtime.h>
#include<iostream>
#include <stdio.h>
#include <stdlib.h>

//Every kernel does the same work of calculating maxSum, sum
//Every thread reads input array always

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int id=threadIdx.x+blockIdx.x*blockDim.x;
    float maxSum = -1e9f;;
    if(id<N){
        float sum=0.0f;
        for(int i=0;i<N;i++){
            maxSum=fmaxf(maxSum,input[i]);
        }
        for(int i=0;i<N;i++){
            sum+=__expf(input[i]-maxSum);
        }
        output[id]=__expf(input[id]-maxSum)/sum;
    }
}


int main(int argc,char* argv[]){
    float *inputCpu,*outputCpu;
    float *inputGpu,*outputGpu;
    // int n=3;
    //argc and argv are runtime parameters
    //argc tells n value
    //argv -> ./softmax.cu 5 10 20 30 44 15
    int n=(argc>1)?atoi(argv[1]):3;
    //Takes value of n from argv which is present at 1st index
    inputCpu = (float*)malloc(n * sizeof(float));
    outputCpu = (float*)malloc(n * sizeof(float));
    for(int i=0;i<n;i++){
        //skips the first 2 parameters of argv array that is ./softmax 5
        inputCpu[i]=(argc>i+2)?atof(argv[i+2]):0.0f;
    }
    // inputCpu[0]=1.0;
    // inputCpu[1]=2.0;
    // inputCpu[2]=3.0;
    // inputCpu[3]=2;
    // inputCpu[4]=6;
    int threadsPerBlock=256;
    int blocksPerGrid=(n+threadsPerBlock-1)/threadsPerBlock;
    

    cudaMalloc(&outputGpu,n*sizeof(float));
    cudaMalloc(&inputGpu,n*sizeof(float));

    cudaMemcpy(outputGpu,outputCpu,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(inputGpu,inputCpu,n*sizeof(float),cudaMemcpyHostToDevice);

    softmax_kernel<<<blocksPerGrid,threadsPerBlock>>>(inputGpu,outputGpu,n);

    cudaMemcpy(outputCpu,outputGpu,n*sizeof(float),cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        printf("%f\n",outputCpu[i]);
    }
    free(inputCpu);
    free(outputCpu);
    cudaFree(inputGpu);
    cudaFree(outputGpu);

    return 0;


}

//to run ./softmax 3 1.0, 2.0, 3.0

//Shared Memory is common between blocks

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================
// KERNEL 1: Naive reduction (interleaved addressing)
// Problem: Warp divergence + bank conflicts
// ============================================================
__global__ void reduceNaive(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// KERNEL 2: Strided indexing (no divergence, but bank conflicts)
// ============================================================
__global__ void reduceStrided(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// KERNEL 3: + First add during global load (2x fewer blocks)
// ============================================================
__global__ void reduceFirstAdd(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load two elements per thread
    int mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n)
        mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// KERNEL 4: + Warp unrolling (no __syncthreads for last warp)
// ============================================================
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceWarpUnroll(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    int mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n)
        mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Last warp — no sync needed
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// KERNEL 5: BEST — Full unroll + warp shuffle (no shared mem!)
// Requires blockSize to be a compile-time constant
// ============================================================
__inline__ __device__ int warpReduceShuffle(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ int blockReduceShuffle(int val) {
    static __shared__ int shared[32]; // One slot per warp
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warpReduceShuffle(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp loads from shared mem
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceShuffle(val);

    return val;
}

__global__ void reduceShuffle(int *g_idata, int *g_odata, unsigned int n) {
    int sum = 0;

    // Grid-stride loop — one kernel handles any input size
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        sum += g_idata[i];
    }

    sum = blockReduceShuffle(sum);

    if (threadIdx.x == 0)
        g_odata[blockIdx.x] = sum;
}

// ============================================================
// HOST: Two-pass reduction launcher
// ============================================================
int reduceCPUFinish(int *d_out, int numBlocks) {
    int *h_out = new int[numBlocks];
    cudaMemcpy(h_out, d_out, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int i = 0; i < numBlocks; i++) sum += h_out[i];
    delete[] h_out;
    return sum;
}

int main() {
    const int N = 1 << 24; // 16M elements
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);

    // Allocate and init host data
    int *h_data = new int[N];
    for (int i = 0; i < N; i++) h_data[i] = 1; // Expected sum = N

    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, NUM_BLOCKS * sizeof(int));
    cudaMemcpy(d_in, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // --- Run best kernel ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduceShuffle<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    int result = reduceCPUFinish(d_out, NUM_BLOCKS);

    printf("Result:   %d (expected %d)\n", result, N);
    printf("Time:     %.3f ms\n", ms);
    printf("Bandwidth: %.1f GB/s\n",
           (N * sizeof(int)) / (ms * 1e-3) / 1e9);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;
    return 0;
}
