#include <stdio.h>
#include <cuda.h>
void printDevProp(struct cudaDeviceProp DevProp){
    /*TODO*/
    printf("%d maximum threads\n", DevProp.maxThreadsDim[1]);
    
    printf("%d number of threads per block\n", DevProp.maxThreadsPerBlock);

    printf("%d number max of threads per multiprocessor\n",DevProp.maxThreadsPerMultiProcessor);

    printf("%d number max of multiprocessor\n",DevProp.multiProcessorCount);
}

int main(){
    int i, devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device #%d\n",devCount);
    for(i=0;i<devCount;i++){
        struct cudaDeviceProp devProp;
        cudaGetDeviceProperties (&devProp,i);
        printDevProp(devProp);
    }
    return(0);
}