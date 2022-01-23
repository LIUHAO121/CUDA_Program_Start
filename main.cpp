
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudafun.h"


int main()
{   
    // cuda_info();
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    //统一内存使用一个托管内存来共同管理host和device中的内存，并且自动在host和device中进行数据传输。
    cudaMallocManaged((void**)&x,nBytes);
    cudaMallocManaged((void**)&y,nBytes);
    cudaMallocManaged((void**)&z,nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    add_run(x,y,z,N);

    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = z[i] - 30.0;
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
 

    matrix_mul_run();
    gpu_add_run();

    return 0;
}