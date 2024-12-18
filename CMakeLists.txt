cmake_minimum_required(VERSION 3.18)

# Project name and enable CUDA
project(CudaProject LANGUAGES CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Read the .env file
file(READ ${CMAKE_SOURCE_DIR}/.env ENV_CONTENT)

# Extract the CUDA architecture value
string(REGEX MATCH "CUDA_ARCH=([0-9]+)" _MATCH ${ENV_CONTENT})

# Extract the matched number (architecture value) from the result
if(_MATCH)
    string(REGEX REPLACE "CUDA_ARCH=" "" CUDA_ARCH ${_MATCH})
else()
    set(CUDA_ARCH 75) # Default value if CUDA_ARCH is not set
endif()

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    main.cu
    include/cpu_eucdist.cu
    include/tree_reduction_eucdist.cu)

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
