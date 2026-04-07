#include<iostream>
#include<cuda_runtime.h>
#define BLOCKSIZE 32

__global__ void matmulkernel(float* A,float* B,float* C,int dim){
    int i,j;
    float temp=0;
    int col=threadIdx.x+blockIdx.x*blockDim.x;
    int row=threadIdx.y+blockIdx.y*blockDim.y;

    __shared__ float Ashared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bshared[BLOCKSIZE][BLOCKSIZE];

    int numTiles=(dim+BLOCKSIZE-1)/BLOCKSIZE;
    for(int tileNum=0;tileNum<numTiles;tileNum++){
        j=tileNum*BLOCKSIZE+threadIdx.x;
        i=tileNum*BLOCKSIZE+threadIdx.y;
        if(row<dim && j<dim)
            Ashared[threadIdx.y][threadIdx.x]=A[row*dim+j];
            // A goes through columns
        else
            Ashared[threadIdx.y][threadIdx.x]=0.0f;
        if(i<dim && col<dim)
            Bshared[threadIdx.y][threadIdx.x]=B[i*dim+col];
            // B goes through rows
        else
            Bshared[threadIdx.y][threadIdx.x]=0.0f;
        // Ashared[threadIdx.y][threadIdx.x]=A[i*dim+j];
        // Bshared[threadIdx.y][threadIdx.x]=B[i*dim+j];

        __syncthreads();

        for(int k=0;k<BLOCKSIZE;k++){
            temp+=Ashared[threadIdx.y][k]*Bshared[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < dim && col < dim)
        C[row * dim + col] = temp;

}
int main(){
    //Declaring 2d as 1d is faster for gpu performance
    int N=4;
    float *Acpu, *Bcpu, *Ccpu;
    float *Agpu, *Bgpu, *Cgpu;

    Acpu=(float *)malloc(N*N*sizeof(float));
    Bcpu=(float *)malloc(N*N*sizeof(float));
    Ccpu=(float *)malloc(N*N*sizeof(float));

    for(int i=0;i<N*N;i++){
        Acpu[i]=sin(i);
        Bcpu[i]=cos(i);
    }
    size_t vectorSize=N*N*sizeof(float);

    cudaMalloc((void **)&Agpu,vectorSize);
    cudaMalloc((void **)&Bgpu,vectorSize);
    cudaMalloc((void **)&Cgpu,vectorSize);

    cudaMemcpy(Agpu,Acpu,vectorSize,cudaMemcpyHostToDevice);
    //Destination, source, size, host->device(cpu->gpu)
    cudaMemcpy(Bgpu,Bcpu,vectorSize,cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCKSIZE,BLOCKSIZE);
    dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    matmulkernel<<<gridDim,blockDim>>>(Agpu,Bgpu,Cgpu,N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float et;

    cudaEventElapsedTime(&et,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(Ccpu,Cgpu,vectorSize,cudaMemcpyDeviceToHost);
    printf("GPU time =%f ms \n",et);
    for(int i=0;i<N*N;i++){
        if((i+1)%N==0)
            printf("%f\n",Ccpu[i]);
        else
            printf("%f ",Ccpu[i]);
    }
    free(Acpu);
    free(Bcpu);
    free(Ccpu);
    cudaFree(Agpu);
    cudaFree(Bgpu);
    cudaFree(Cgpu);

    return 0;
}