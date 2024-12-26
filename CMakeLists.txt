cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RayTracing LANGUAGES CXX CUDA)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA Architectures (adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Option to enable or disable OpenGL
option(USE_OPENGL "Enable OpenGL support" ON)

# Find Packages
find_package(CUDAToolkit REQUIRED)

if(USE_OPENGL)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 3.3 REQUIRED)
    find_package(GLEW REQUIRED)
endif()

# Source Files
set(SOURCES
    main.cu
)

# Header Files
set(HEADERS
    camera.h
    hitable.h
    hitable_list.h
    material.h
    ray.h
    sphere.h
    vec3.h
)

# Create Executable Target
add_executable(RayTracing ${SOURCES} ${HEADERS} "acceleration_structure.h")

# Include Directories
target_include_directories(RayTracing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(USE_OPENGL)
    target_include_directories(RayTracing PRIVATE
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIRS}
    )
    target_compile_definitions(RayTracing PRIVATE USE_OPENGL)
endif()

# Link Libraries
target_link_libraries(RayTracing PRIVATE
    CUDA::cudart
    CUDA::curand
)

if(USE_OPENGL)
    target_link_libraries(RayTracing PRIVATE
        ${OPENGL_LIBRARIES}
        glfw
        ${GLEW_LIBRARIES}
    )
endif()

# Set CUDA Compilation Properties
set_target_properties(RayTracing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    # Ensure the linker uses the correct language
    LINKER_LANGUAGE CUDA
)
