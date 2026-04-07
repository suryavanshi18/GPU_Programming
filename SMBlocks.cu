#include<iostream>
#include<cuda_runtime.h>
using namespace std;

__global__ void sm_roll_call(){
    const int tid=threadIdx.x;
    uint streamingMultiprocessorId;
    asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId));
    //Move the SM ID (%smid) into a register (%0)

    printf("Thread %d running on SM %d!\n", tid, streamingMultiprocessorId);
    //cout doesnt work in kernel
}
int main(){
    sm_roll_call<<<4, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}