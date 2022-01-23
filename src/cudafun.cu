
#include <iostream>
#include "cudafun.h"

#define BLOCK_SIZE  32

#define CHECK(x) check_runtime(x)

// CUDA 不会自动报错，需要check
void check_runtime(cudaError_t e){
        if (e != cudaSuccess) {
            std::cout << "File:          "<< __FILE__ << std::endl;
            std::cout << "Line:          "<< __LINE__<< std::endl;
            std::cout << "Error Name:    "<< cudaGetErrorName(e) << std::endl;
            std::cout << "Error String:    "<< cudaGetErrorString(e) << std::endl;
        }
    }


void cuda_info()
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}


// 原子操作，实现向量求和
__global__ void gpu_add(int* input,int count,int* res)
{
    __shared__ int bowan[BLOCK_SIZE];  // 在block 内申请共享内存，并冲global memory复制数据
    int temp = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;idx < count;idx += gridDim.x * blockDim.x)
    {
        temp += input[idx];
    }
    bowan[threadIdx.x] = temp;
    __syncthreads(); // 等待block内的所有线程完成复制
    
    for (int length = BLOCK_SIZE/2;length >= 1;length /=2)
    {
        int double_add = -1;
        if(threadIdx.x < length)
        {
            double_add = bowan[threadIdx.x] + bowan[threadIdx.x + length];
        }
        __syncthreads();
        if (threadIdx.x < length)
        {
            bowan[threadIdx.x] = double_add;
        }
        __syncthreads();
    }
    if (blockDim.x * blockIdx.x < count)
    {
        if (threadIdx.x ==0)
        {
            atomicAdd(res,bowan[0]);
        }
    }
}

void gpu_add_invoke(int* input,int count,int* res)
{
    int grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_size);
    dim3 dimBlock(BLOCK_SIZE);
    gpu_add<<<dimGrid, dimBlock>>>(input, count, res);
}


void gpu_add_run()
{
    int count = 100;
    int* h_input;
    int* h_output;
    cudaMallocHost((void**)&h_input,sizeof(int)*count);
    cudaMallocHost((void**)&h_output,sizeof(int));
    for (int i=0;i<count;i++)
    {
        h_input[i] = i+1;
    }

    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input,sizeof(int)*count);
    cudaMalloc((void**)&d_output,sizeof(int));

    cudaMemcpy(d_input,h_input,sizeof(int)*count,cudaMemcpyHostToDevice);
    gpu_add_invoke(d_input,count,d_output);
    cudaMemcpy(h_output,d_output,sizeof(int),cudaMemcpyDeviceToHost);
    
    std::cout << "add result is " << *h_output<< std::endl;
}



// 矩阵乘法
static __global__ void matrix_mul(int* m,int* n,int m_size,int n_size,int k_size,int* res)
{
    // m shape = (m_size,n_size)
    // n shape = (n_size,k_size)
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < k_size && row < m_size) //一定要加范围限制，并不是所有的线程都会用到，只有在范围内的线程才可以
    {
        int sum = 0;
        for (int i=0;i<n_size;i++)
        {
            sum += m[row * n_size + i] * n[i * k_size + col];
        }
        res[row*k_size + col] = sum;
    }
}

void matrix_mul_invoker(int* m,int* n,int m_size,int n_size,int k_size,int* res)
{
    unsigned int grid_rows = (m_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    matrix_mul<<<dimGrid,dimBlock>>> (m,n,m_size,n_size,k_size,res);
}

void matrix_mul_run()
{
    int m_size = 100;
    int n_size = 100;
    int k_size = 100;
    // m shape = (m_size,n_size)
    // n shape = (n_size,k_size)
    int* h_a;
    int* h_b;
    int* h_c;
    CHECK(cudaMallocHost((void**)&h_a, sizeof(int)*m_size*n_size));
    CHECK(cudaMallocHost((void**)&h_b, sizeof(int)*n_size*k_size));
    CHECK(cudaMallocHost((void**)&h_c, sizeof(int)*m_size*k_size));

    for (int i =0 ; i< m_size; i++)
    {
        for (int j=0; j< n_size;j++)
        {
            h_a[i*n_size + j] = 2;
        }
    }

    for (int i =0 ; i< n_size; i++)
    {
        for (int j=0; j< k_size;j++)
        {
            h_b[i*k_size + j] = 3;
        }
    }

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a,sizeof(int)*m_size*n_size);
    cudaMalloc((void**)&d_b,sizeof(int)*n_size*k_size);
    cudaMalloc((void**)&d_c,sizeof(int)*m_size*k_size);
    

    // copy matrix frow host to device
    cudaMemcpy(d_a,h_a,sizeof(int)*m_size*n_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,sizeof(int)*n_size*k_size,cudaMemcpyHostToDevice);

    matrix_mul_invoker(d_a,d_b,m_size,n_size,k_size,d_c);
    cudaMemcpy(h_c,d_c,sizeof(int)*m_size*k_size,cudaMemcpyDeviceToHost);

    for(int i=0;i<3;i++)
    {
        std::cout << "matrix mul res" << std::endl;
        std::cout << h_c[i] << std::endl;
    }

}



static __global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

// 不要在cpp文件中include以 .cu 文件，因为遇到<<< >>> 这样的符号容易编译不过
// 一般将所有的 .cu 生成动态库，然后再连接该库

void add_run(float* x, float * y, float* z, int n)
{
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    add<<<gridSize, blockSize>>>(x, y, z, n);
}