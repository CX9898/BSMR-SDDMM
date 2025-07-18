# Specify the minimum version of cmake
cmake_minimum_required(VERSION 3.26)

# Project Information
project(Sputnik-sddmm)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# Specifies which GPU architectures are supported when compiling CUDA code (optional)
#set(CMAKE_CUDA_ARCHITECTURES 80)

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

# Add header directory (locally)
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

find_package(glog REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glog::glog)

add_subdirectory(${INCLUDE_DIR}/abseil-cpp) # 👈 新增这一行
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        absl::random_random
        absl::random_distributions
        absl::random_internal_randen
        absl::random_seed_sequences
        absl::status
        absl::optional
        absl::flat_hash_map
        absl::strings
        absl::time
)

# Linked cuda Runtime library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
