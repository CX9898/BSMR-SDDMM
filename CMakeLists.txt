# Specify the minimum version of cmake
cmake_minimum_required(VERSION 3.26)

# Project Information
project(BSMR-sddmm)

# Add languages needed when building the project
enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# Sets the C++ standard used by the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Specifies which GPU architectures are supported when compiling CUDA code (optional)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Save the folder path in a variable
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# All source files in src folder are stored in SRC_FILES variable
file(GLOB SRC_FILES "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cc" "${SRC_DIR}/*.cxx" "${SRC_DIR}/*.cu")

# Output file list information
message(STATUS "Src files: ${SRC_FILES}")

# Add generate target
add_executable(${PROJECT_NAME})

## Print kernel function register information
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set the CUDA separable compilation property
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the source file to the build target
target_sources(${PROJECT_NAME} PRIVATE ${SRC_FILES})

# Set the installation Path
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

# Set installation rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Add header directory (locally)
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# Linked cuda Runtime library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Linked cuBLAS library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)

# Linked cuFFT library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cufft)

# Linked cuRAND library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::curand)

# Linked cuSOLVER library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusolver)

# Linked cuSPARSE library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusparse)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Linked OpenMP library
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
