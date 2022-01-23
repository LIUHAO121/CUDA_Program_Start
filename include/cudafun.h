// 先给出函数声明，保证该文件能正常编译, 不要直接include cudafun.cu,编译不会通过
// 将所有的cu文件生成一个动态库
// 先声明在cudafun.cu要使用的函数, 在开始link时,连接器会在动态库里找该函数的定义
void add_run(float* x, float * y, float* z, int n); 
void matrix_mul_run();
void gpu_add_run();
void vec_add_run();
void vec_add_stream_run();