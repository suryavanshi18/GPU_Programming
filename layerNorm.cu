#include<iostream>
#include<cuda_runtime.h>
using namespace std;

struct Tensor{
    int rows;
    int cols;
    float *data;
    __device__ int size(){
        return rows*cols;
    }
    __device__ float get(const int i,const int j){
        return data[i*cols+j];
    }
    __device__ void set(const int i,const int j,const float val){
        data[i*cols+j]=val;
    }
};

__global__ void layerNorm(Tensor* A){
    extern __shared__ float sharedMemory[];
    float* mean=sharedMemory;
    float* variance=sharedMemory+A->rows;
    float* invstdev=sharedMemory+2*A->rows;

    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<A->rows){
        mean[tid]=0.0f;
        for(int j=0;j<A->cols;j++){
            mean[tid]+=A->get(tid,j);
        }
        mean[tid]/=A->cols;
    }

    __syncthreads();

    if(tid<A->rows){
        variance[tid]=0.0f;
        for(int j=0;j<A->cols;j++){
            float diff=(A->get(tid,j)-mean[tid]);
            variance[tid]+=diff*diff;
        }
        variance[tid]/=A->cols;
        invstdev[tid]=rsqrtf(variance[tid]+1e-5);
    }
    __syncthreads();
    if(tid<A->rows){
        for(int j=0;j<A->cols;j++){
            float normalized=(A->get(tid,j)-mean[tid])*invstdev[tid];
            A->set(tid,j,normalized);
        }
    }
}

int main(){
    const int rows = 2;
    const int cols = 3;
    const int BLOCKSIZE = 16;
    const int tensorSize = rows * cols;
    const int size = rows * cols * sizeof(float);
    // Total threads=threadsPerBlock*Number of blocks
    // BlockDim must be multiple of 32
    // gridDim-> calculate based on your data-> totalElements+blockSize-1/blockSize
    dim3 blockDim(16*16);
    dim3 gridDim((tensorSize+BLOCKSIZE-1)/BLOCKSIZE);
    size_t sharedMemory =3*rows*sizeof(float);
    // mean,variance,invstdev

    float h_data[tensorSize]={5.0f,1.5f,2.0f,3.0f, 4.0f, 1.0f};
    float* d_data;

    cudaMalloc(&d_data,size);
    cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice);
    Tensor h_tensor;
    h_tensor.data=d_data;
    h_tensor.rows=rows;
    h_tensor.cols=cols;

    Tensor* d_tensor;
    cudaMalloc(&d_tensor,sizeof(Tensor));
    cudaMemcpy(d_tensor,&h_tensor,sizeof(Tensor),cudaMemcpyHostToDevice);


    layerNorm<<<gridDim,blockDim,sharedMemory>>>(d_tensor);
    //Without the third argument the default allocation is 0 bytes. Your threads access memory that was never allocated.
    cudaDeviceSynchronize();
    cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            cout<<h_data[i*cols+j]<<" ";
        }
        cout<<endl;
    }
    cudaFree(d_data);
    cudaFree(d_tensor);
    return 0;
}