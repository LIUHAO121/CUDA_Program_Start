
#include <iostream>
#include "cudafun.h"

#define BLOCK_SIZE  32
#define TILE_SIZE 16
#define CHECK(x) check_runtime(x)

// 不要在cpp文件中include以 .cu 文件，因为遇到<<< >>> 这样的符号容易编译不过
// 一般将所有的 .cu 生成动态库，然后再连接该库

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
    std::cout << "-------------- gpu_add_run -------------" << std::endl;
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
    std::cout << "-------------- matrix_mul_run -------------" << std::endl;
    int m_size = 100;
    int n_size = 150;
    int k_size = 1000;
    // m shape = (m_size,n_size)
    // n shape = (n_size,k_size)
    int* h_a;
    int* h_b;
    int* h_c;
    
    h_a = (int*)malloc(sizeof(int)*m_size*n_size);
    h_b = (int*)malloc(sizeof(int)*n_size*k_size);
    h_c = (int*)malloc(sizeof(int)*m_size*k_size);

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
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,sizeof(int)*m_size*k_size,cudaMemcpyDeviceToHost);

    std::cout << "h_out first element " << h_c[0] << std::endl;
    std::cout << "h_out last element " << h_c[m_size * k_size - 1] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a); // 不能释放由cudaMallocHost分配的内存
    free(h_b);  
    free(h_c);
}




// 通过share memory 进行矩阵乘法
static __global__ void share_matrix_mul(float* M,float* N, int height,int k, int width, float* res)
{
    // matrix M (h * k)
    // matrix N (k * w)
    __shared__ float sm[TILE_SIZE][TILE_SIZE];
    __shared__ float sn[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // blockDim.y = blockDim.x = TILE_SIZE
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float pvalue = 0;

    for (int i = 0;i < (k + TILE_SIZE - 1)/TILE_SIZE;i++)
    {
        if (row < height && i * TILE_SIZE + tx < k)
            sm[ty][tx] = M[row * k + i * TILE_SIZE + tx];
        else
            sm[ty][tx] = 0;
        if (col < width && i * TILE_SIZE + ty < k)
            sn[ty][tx] = N[(i * TILE_SIZE + ty) * width + col];
        else
            sn[ty][tx] = 0;
        __syncthreads();

        for(int j=0;j<TILE_SIZE;j++)
        {
            pvalue += sm[ty][j] * sn[j][tx];
        }
        __syncthreads();

    }
    if (row < height && col < width)
        res[row * width + col] = pvalue;
}

void share_matrix_invoke(float* M,float* N, int height,int k, int width, float* res)
{
    unsigned int grid_rows = (height + TILE_SIZE - 1) / TILE_SIZE;
    unsigned int grid_cols = (width + TILE_SIZE - 1) / TILE_SIZE;
    dim3 dimBlock(TILE_SIZE,TILE_SIZE);
    dim3 dimGrid{grid_cols,grid_rows};
    share_matrix_mul<<<dimGrid,dimBlock>>>(M,N,height,k,width,res);
}

void share_matrix_mul_run()
{
    std::cout << "-------------- share_matrix_mul_run -------------" << std::endl;
    int m_size = 15;
    int k_size = 101;
    int n_size = 15;

    // h_m shape = (m_size,k_size)
    // h_n shape = (k_size,n_size)

    float* h_m;
    float* h_n;
    float* h_out;
    
    h_m = (float*)malloc(sizeof(float) * m_size * k_size);
    h_n = (float*)malloc(sizeof(float) * k_size * n_size);
    h_out = (float*)malloc(sizeof(float) * m_size * n_size);

    for (int i =0 ; i< m_size; i++)
    {
        for (int j=0; j< k_size;j++)
        {
            h_m[i*k_size + j] = 2;
        }
    }

    for (int i =0 ; i < k_size; i++)
    {
        for (int j=0; j < n_size;j++)
        {
            h_n[i * n_size + j] = 3;
        }
    }

    float* d_m;
    float* d_n;
    float* d_out;
    cudaMalloc((void**)&d_m,sizeof(float)*m_size*k_size);
    cudaMalloc((void**)&d_n,sizeof(float)*k_size*n_size);
    cudaMalloc((void**)&d_out,sizeof(float)*m_size*n_size);

    cudaMemcpy(d_m,h_m,sizeof(float)*m_size*k_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_n,h_n,sizeof(float)*k_size*n_size,cudaMemcpyHostToDevice);

    share_matrix_invoke(d_m,d_n,m_size,k_size,n_size,d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,sizeof(int)*m_size*n_size,cudaMemcpyDeviceToHost);
    
    std::cout << "h_out first element " << h_out[0] << std::endl;
    std::cout << "h_out second element " << h_out[1] << std::endl;
    std::cout << "h_out n_size-1 element " << h_out[2*n_size-1] << std::endl;
    std::cout << "h_out n_size element " << h_out[2*n_size] << std::endl;
    std::cout << "h_out n_size+1 element " << h_out[2*n_size + 1] << std::endl;
    std::cout << "h_out last element " << h_out[m_size * n_size - 1] << std::endl;

    // cudaFree(d_m);
    // cudaFree(d_n);
    // cudaFree(d_out);
    // free(h_m); // 不能释放由cudaMallocHost分配的内存
    // free(h_n);  
    // free(h_out);
}


static __global__ void vec_add(float* x, float * y, float* z, int n)
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

template <typename T>
void vec_add_invoke(T* x, T* y, T* z, int n)
{
    dim3 blockSize(32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    vec_add<<<gridSize, blockSize>>>(x, y, z, n);
}

void vec_add_run()
{
    std::cout << "-------------- vec_add_run -------------" << std::endl;
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

    vec_add_invoke(x,y,z,N);

    //等待gpu执行结束,
    //kernel执行是与host异步的，由于托管内存自动进行数据传输，这里要用cudaDeviceSynchronize()函数保证device和host同步，这样后面才可以正确访问kernel计算的结果。
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
}


void vec_add_stream_invoke(float* x, float * y, float* z, int n,cudaStream_t stream)
{
    dim3 blockSize(32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    vec_add<<<gridSize, blockSize,0,stream>>>(x, y, z, n);
}

void vec_add_stream_run()
{
    std::cout << "-------------- vec_add_stream_run -------------" << std::endl;
    std::cout << "-------------- 使用cuda流来加速应用程序 -------------" << std::endl;
    int n=1024;
    int nBytes = n * sizeof(float);
    // 申请host内存
    float *h_x, *h_y, *h_z;
    cudaMallocHost((void**)&h_x,nBytes);
    cudaMallocHost((void**)&h_y,nBytes);
    cudaMallocHost((void**)&h_z,nBytes);

    for (int i = 0; i < n;i++)
    {
        h_x[i] = 10.0;
        h_y[i] = 20.0;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x,nBytes);
    cudaMalloc((void**)&d_y,nBytes);
    cudaMalloc((void**)&d_z,nBytes);
    int nstream = 2;
    cudaStream_t streams[nstream];
    for (int i = 0;i<nstream;i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    int eles_per_stream = n/nstream;
    for (int i=0;i<nstream;i++)
    {
        int offset = i * eles_per_stream;
        cudaMemcpyAsync(&d_x[offset], &h_x[offset], eles_per_stream*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_y[offset], &h_y[offset], eles_per_stream*sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        vec_add_stream_invoke(&d_x[offset], &d_y[offset], &d_z[offset], eles_per_stream,streams[i]);
        cudaMemcpyAsync(&h_z[offset], &d_z[offset], eles_per_stream*sizeof(float),cudaMemcpyDeviceToHost,streams[i]);
    }
    for (int i=0;i<nstream;i++)
    {
        cudaStreamSynchronize(streams[i]);
    }

    for (int i=0;i<nstream;i++)
    {
        std::cout << "stream res at position "<< i*eles_per_stream << " = " <<h_z[i*eles_per_stream] << std::endl;
    }
    

}