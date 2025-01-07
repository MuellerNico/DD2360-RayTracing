# DD2360-RayTracing 

A GPU-accelerated ray tracer implemented in CUDA C++, featuring an octree acceleration structure for optimized scene traversal.

Based on: [Ray Tracing in One Weekend](https://github.com/rogerallen/raytracinginoneweekendincuda) by [Roger Allan](https://github.com/rogerallen). 

Project work for DD2360 Applied GPU Programming at KTH Stockholm.

## Run in Google Colab: Quick Start

1. Open the notebook `raytracing.ipynb` in Google Colab
2. Follow the steps there

## Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- GLFW3 and GLEW (only if using OpenGL display)
- CUDA-capable GPU with compute capability 7.5+

## Usage Details

### Output modes
`$./RayTracing [mode]`
- 0: Output image to stdout (default)
- 1: Disable image output
- 2: Render in OpenGL window (if compiled with USE_OPENGL)
- 3: Output image to file (creates output.ppm)

### Parameters

- `#define USE_OCTREE`: Comment/uncomment for octree acceleration structure (main.cu)
- `NUM_SPHERES`: adjust number of spheres in scene (main.cu)
- When changing the number of spheres, one might have to adjust `SPHERES_PER_LEAF`to increase the tree's capacity. This value is limited by memory availability (acceleration_structure.h)
- `#define USE_FP16`: Comment/uncomment for reduced 16bit precicion (precision_types.h)

### Progressive refinement preview
Use `$ cmake -DUSE_OPENGL=OFF` to disable the live preview window. This option is necessary when running in Colab or anywhere else where OpenGL is not supported. 

## Project Structure

- main.cu: Entry point and rendering logic
- acceleration_structure.h: Octree implementation for spatial partitioning
- precision_types.h: Precision control (FP32/FP16)
- camera.h: Camera model with depth of field support
- hitable.h: Abstract base class for ray-intersectable objects
- material.h: Material definitions (Lambertian, metal, dielectric)
- sphere.h: Sphere primitive implementation
- vec3.h: Vector mathematics

Only the first three files contain changes relevant to this project. The rest remains the same as in the [base implementation](https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ch12_where_next_cuda).



## Contributors
- Christiane Kobalt
- Nicolas Müller
- Maximilian Ranzinger
- Robin Sieber
