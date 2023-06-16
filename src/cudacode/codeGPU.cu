#include "codeGPU.h"
#define THR_PER_BLOCK 1024 

__global__ void cudaGlobal_matrix_dot (double* A, double* B, double* C, int a_rows, int a_columns, int b_columns){
    int row_GPU= blockIdx.y * blockDim.y + threadIdx.y;
    int column_GPU= blockIdx.x * blockDim.x+ threadIdx.x;

    
    if(row_GPU< a_rows && column_GPU < b_columns){
        double sum=0.0;
        for (int i=0; i<a_columns;i++){
            sum+= A[row_GPU* a_columns +i] * B[i* b_columns + column_GPU];
        }
        C[row_GPU*b_columns+col]=sum;  
    }
}

__global__ void cudaGlobal_matrix_add (double* A, double* B, double* C, int a_rows, int a_columns){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < a_rows && j < a_columns){       
        C[i*a_columns+ j] = A[i * a_columns + j] + B[i * a_columns + j];
    }
}

void matrix_mul_addGPU(double *c, double *a, double *b, int a_rows, int a_columns, int b_rows, int b_columns, double *d){
    size_t NA=a_rows*a_columns;
    size_t NB=b_rows*b_columns;
    size_t NC=a_rows*b_columns;
    size_t ND=a_rows*b_columns; // a x b + c x d = a x d

    //GPU-ra erabiliko dituen aldagaiak
    double *GPU_A, *GPU_B, *GPU_C, *GPU_D;

    gpuErrchk(cudaMalloc(&GPU_A, NA*sizeof(double)));
    gpuErrchk(cudaMalloc(&GPU_B, NB*sizeof(double)));
    gpuErrchk(cudaMalloc(&GPU_C, NC*sizeof(double)));
    gpuErrchk(cudaMalloc(&GPU_D, ND*sizeof(double)));

    gpuErrchk(cudaMemcpy(GPU_A,a,NA*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(GPU_B,b,NB*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(GPU_D,d,ND*sizeof(double), cudaMemcpyHostToDevice));

    //Emaitza gordetzeko balioko du beraz hasieratu memorian
    gpuErrchk(cudaMemset(GPU_C,c,NC*sizeof(float)));

    dim3 thr_per_blk;// hari kopuru bloke bakoitzean 
    dim3 blk_in_grid; // zenbat bloke grid bakoitzean

    thr_per_blk=dim3(32,32);
    blk_in_grid= dim3(ceil((double)NA/thr_per_blk.x),ceil((double)NA/thr_per_blk.y));
    //matrizeak biderkatu
    cudaGlobal_matrix_dot<<blk_in_grid,thr_per_blk>>(GPU_A,GPU_B,GPU_C, a_rows,a_columns, b_columns);

    cudaFree(GPU_A);
    cudaFree(GPU_B);
    // Matrizeak gehitu
    cudaGlobal_matrix_add<<blk_in_grid,thr_per_blk>>(GPU_C,GPU_D,GPU_C,a_rows,b_columns);

    // GPU memoriatik host-era bueltatu
    gpuErrchk(cudaMemcpy(c,GPU_C,NC*sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaFree(GPU_C);
    cudaFree(GPU_D);
}