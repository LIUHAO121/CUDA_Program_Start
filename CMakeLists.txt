cmake_minimum_required(VERSION  3.9 FATAL_ERROR)
project(main CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS "-g -G -O3 -Wno-deprecated-gpu-targets")


set(CUDA_DIR  "/usr/local/cuda/")
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)


file(GLOB cuda_srcs ${PROJECT_SOURCE_DIR}/*.cu)
add_library(plugin_list SHARED ${cuda_srcs})
file(GLOB cpp_srcs ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(main ${cpp_srcs})

target_link_libraries(main cuda cublas cudart cudnn)
target_link_libraries(main plugin_list)