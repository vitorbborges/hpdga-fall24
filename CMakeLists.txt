cmake_minimum_required(VERSION 3.18)

# Project name and enable CUDA
project(CudaProject LANGUAGES CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)


# Include directories
include_directories(include)

# Source files
set(SOURCES
    main.cu
)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Custom target to build and run
add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the compiled executable"
)
