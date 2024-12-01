cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RayTracing LANGUAGES CXX CUDA)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA Architectures (adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find Packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

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
add_executable(RayTracing ${SOURCES} ${HEADERS})

# Include Directories
target_include_directories(RayTracing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}  # Include GLEW headers
)

# Link Libraries
target_link_libraries(RayTracing PRIVATE
    CUDA::cudart
    CUDA::curand
    OpenGL::GL
    glfw
    ${GLEW_LIBRARIES}
)

# Set CUDA Compilation Properties
set_target_properties(RayTracing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    # Ensure the linker uses the correct language
    LINKER_LANGUAGE CUDA
)

# Handle Windows-specific flags
if(MSVC)
    # Remove conflicting flags
    foreach(FLAG_VAR CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS_MINSIZEREL
                     CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
        string(REPLACE "/MDd" "" ${FLAG_VAR} "${${FLAG_VAR}}")
        string(REPLACE "/MD" "" ${FLAG_VAR} "${${FLAG_VAR}}")
        string(REPLACE "/ZI" "/Z7" ${FLAG_VAR} "${${FLAG_VAR}}")
    endforeach()

    # Add necessary compiler options
    target_compile_options(RayTracing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --compiler-options /FS
    >)
endif()
