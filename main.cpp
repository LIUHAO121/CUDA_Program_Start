
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudafun.h"


int main()
{   
    // cuda_info();
    
    vec_add_run();
    matrix_mul_run();
    gpu_add_run();
    vec_add_stream_run();
    share_matrix_mul_run();
    return 0;
}