#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include <string>
#include <vector>
#include "precision_types.h"
#include "acceleration_structure.h"

#ifdef USE_OPENGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#define NUM_SPHERES (22 * 22 + 1 + 3)

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
#define USE_OCTREE
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, Octree* d_octree, sphere(*d_list)[NUM_SPHERES]) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
#ifdef USE_OCTREE
        Octhit* octhit = hitTree(d_octree, r);
		// debug
        // printf("Octree hit: %d\n", octhit->num_p_hits);
		hitable_list* world_list = (hitable_list*)(*world);
		hit_record temp_rec, closest_rec;
		bool hit_anything = false;
		real_t closest_so_far = FLT_MAX;
		for(int j = 0; j < octhit->num_p_hits; j++)
		{
			int sphere_idx = octhit->possible_hits[j];
			if(sphere_idx < NUM_SPHERES) {
				if(world_list->list[sphere_idx]->hit(cur_ray, real_t(0.001f), closest_so_far, temp_rec))
				{
					hit_anything = true;
					closest_so_far = temp_rec.t;
					closest_rec = temp_rec;
				}
			}
		}
		if(hit_anything) {
            ray scattered;
            vec3 attenuation;
            if (closest_rec.mat_ptr->scatter(cur_ray, closest_rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }

#else
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
#endif
        else {
            const vec3 unit_direction = unit_vector(cur_ray.direction());
            const real_t t = real_t(0.5f) * (unit_direction.y() + (real_t)1.0f);
            const vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	// Original: Each thread gets same seed, a different sequence number, no offset
	// curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
	// BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
	// performance improvement of about 2x!
	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state, Octree* d_octree, sphere(*d_list)[NUM_SPHERES]) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
		real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state, d_octree, d_list);
	}
	
	rand_state[pixel_index] = local_rand_state;
	col /= real_t(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
	
}

__global__ void render_progressive(vec3* fb, int max_x, int max_y, int current_sample, camera** cam, hitable** world, curandState* rand_state, Octree* d_octree, sphere(*d_list)[NUM_SPHERES]) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);

	// Render one sample per pixel per call
	real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
	real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);
	ray r = (*cam)->get_ray(u, v, &local_rand_state);
	col = color(r, world, &local_rand_state, d_octree, d_list);

	rand_state[pixel_index] = local_rand_state;

	// Accumulate color
	if (current_sample == 1) {
		fb[pixel_index] = col;
	}
	else {
		fb[pixel_index] += col;
	}
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(sphere (*d_list)[NUM_SPHERES], hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		(*d_list)[0] = sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(vec3(0.5, 0.5, 0.5)));	// ground plane as sphere
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				const real_t choose_mat = RND;
				const vec3 center(a + RND, 0.2, b + RND);
				if (choose_mat < real_t(0.8f)) {
					(*d_list)[i++] = sphere(center, 0.2,
						new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
				}
				else if (choose_mat < real_t(0.95f)) {
					(*d_list)[i++] = sphere(center, 0.2,
						new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
				else {
					(*d_list)[i++] = sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		(*d_list)[i++] = sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		(*d_list)[i++] = sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		(*d_list)[i++] = sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		*rand_state = local_rand_state;
		int num_hitables = 22 * 22 + 1 + 3;
		hitable** d_hitable = new hitable*[num_hitables];	// convert to array of pointers to keep changes minimal
		d_hitable[0] = &((*d_list)[0]);
		for(int i = 1; i < num_hitables; i++)
		{
			d_hitable[i] = &((*d_list)[i]);
		}
		*d_world = new hitable_list(d_hitable, num_hitables );

		const vec3 lookfrom(13, 2, 3);
		const vec3 lookat(0, 0, 0);
		const real_t dist_to_focus = 10.0; (lookfrom - lookat).length();
		const real_t aperture = 0.1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			30.0,
			real_t(nx) / real_t(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(sphere(*d_list)[NUM_SPHERES], hitable** d_world, camera** d_camera) {
	for (int i = 0; i < NUM_SPHERES; i++) {
		delete (*d_list[i]).mat_ptr;
	}
	delete d_list;
	delete* d_world;
	delete* d_camera;
}

#ifdef USE_OPENGL
int render_in_window(const int nx, const int ny, dim3 blocks, dim3 threads, vec3* fb, camera** d_camera, hitable** d_world, curandState* d_rand_state, Octree* d_octree, sphere(*d_list)[NUM_SPHERES])
{
	const int num_pixels = nx * ny;

	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Error: GLFW initialization failed.\n";
		return -1;
	}

	// Create a GLFW window
	GLFWwindow* window = glfwCreateWindow(nx, ny, "CUDA Ray Tracer", nullptr, nullptr);
	if (!window) {
		std::cerr << "Error: Window creation failed.\n";
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW (necessary to get OpenGL extensions)
	glewExperimental = GL_TRUE;
	const GLenum glew_status = glewInit();
	// Ignore GL_INVALID_ENUM error caused by glewInit()
	glGetError();
	if (glew_status != GLEW_OK) {
		std::cerr << "Error: GLEW initialization failed.\n";
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Create OpenGL texture
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	// Allocate texture storage
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, nx, ny, 0, GL_RGB, GL_FLOAT, NULL);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	int current_sample = 0;
	const int max_samples = 4000; // Set the number of samples per pixel

	// Main render loop
	while (!glfwWindowShouldClose(window) && current_sample < max_samples) {
		glfwPollEvents();

		current_sample++;

		// Launch render kernel
		render_progressive << <blocks, threads >> > (fb, nx, ny, current_sample, d_camera, d_world, d_rand_state, d_octree, d_list);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Copy data to OpenGL texture
		// Apply gamma correction and averaging
		std::vector<real_t> pixels(num_pixels * 3);
		for (int i = 0; i < num_pixels; ++i) {
			vec3 col = fb[i] / real_t(current_sample);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2])); // Gamma correction
			pixels[i * 3 + 0] = col.r();
			pixels[i * 3 + 1] = col.g();
			pixels[i * 3 + 2] = col.b();
		}

		// Update texture
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGB, GL_FLOAT, pixels.data());

		// Render textured quad
		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
			glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
			glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
			glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
		}
		glEnd();
		glDisable(GL_TEXTURE_2D);

		glfwSwapBuffers(window);

		// Optional: Display progress
		std::cout << "Sample " << current_sample << "/" << max_samples << "\r";
		std::cout.flush();
	}

	// Terminate GLFW
	glfwDestroyWindow(window);
	glfwTerminate();
}
#endif

void output_to_stream(std::ostream& ostream, const int nx, const int ny, const vec3* fb)
{
	ostream << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			const size_t pixel_index = j * nx + i;
			const int ir = static_cast<int>(255.99 * fb[pixel_index].r());
			const int ig = static_cast<int>(255.99 * fb[pixel_index].g());
			const int ib = static_cast<int>(255.99 * fb[pixel_index].b());
			ostream << ir << " " << ig << " " << ib << "\n";
		}
	}
}

void output_to_console(const int nx, const int ny, const vec3* fb)
{
	output_to_stream(std::cout, nx, ny, fb);
}

void output_to_file(const int nx, const int ny, const vec3* fb)
{
	std::ofstream outfile("output.ppm");
	output_to_stream(outfile, nx, ny, fb);
	outfile.close();
}

int main(int argc, char** argv) {
	const int nx = 1200;
	const int ny = 800;
	const int ns = 10;
	const int tx = 8;
	const int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int output_mode = 0; // 0 = to stdout (default), 1 = disabled, 2 = to window, 3 = to file
	if (argc > 1) {
		output_mode = std::stoi(argv[1]);
	}
	std::cerr << "Output mode: " << output_mode << "\n";

	const int num_pixels = nx * ny;
	const size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb), fb_size));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_rand_state), num_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_rand_state2), 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	sphere (*d_list)[NUM_SPHERES];
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_list), NUM_SPHERES * sizeof(sphere)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(hitable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(camera*)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// copy spheres to CPU to fill into octree
	// first copy the addresses on the GPU
	sphere* cpu_spheres = static_cast<sphere*>(malloc(sizeof(sphere) * NUM_SPHERES));
	checkCudaErrors(cudaMemcpy(cpu_spheres, d_list, NUM_SPHERES * sizeof(sphere), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	// build octree
	Octree* octree = buildOctree(cpu_spheres, NUM_SPHERES);

	// upload octree to gpu
	Octree* d_octree;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_octree),sizeof(Octree)));
	checkCudaErrors(cudaMemcpy(d_octree, octree, sizeof(Octree), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state, d_octree, d_list);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	const double timer_seconds = static_cast<double>(stop - start) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output methods
	switch (output_mode)
	{
	case 0:
		output_to_console(nx, ny, fb);
		break;
	case 1:
		// do nothing
		break;
#ifdef USE_OPENGL
	case 2:
		render_in_window(nx, ny, blocks, threads, fb, d_camera, d_world, d_rand_state, d_octree, d_list);
		break;
#endif
	case 3:
		output_to_file(nx, ny, fb);
	default:
		// do nothing
		break;
	}

	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_rand_state2));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}