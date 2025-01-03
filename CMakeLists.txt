cmake_minimum_required(VERSION 3.18)

# Project name and enable CUDA
project(CudaProject LANGUAGES CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Debugging: Print the source directory
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# Check if the .env file exists
if(EXISTS "${CMAKE_SOURCE_DIR}/.env")
    message(STATUS ".env file found at ${CMAKE_SOURCE_DIR}/.env")
    file(READ "${CMAKE_SOURCE_DIR}/.env" ENV_CONTENT)
    string(REGEX MATCH "CUDA_ARCH=([0-9]+)" _MATCH ${ENV_CONTENT})
    if(_MATCH)
        string(REGEX REPLACE "CUDA_ARCH=" "" CUDA_ARCH ${_MATCH})
    else()
        message(WARNING "CUDA_ARCH not found in .env file. Using default architecture 75.")
        set(CUDA_ARCH 75)
    endif()
else()
    message(WARNING ".env file not found. Using default CUDA_ARCH=75.")
    set(CUDA_ARCH 75) # Default value
endif()

# Debugging messages
message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

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
