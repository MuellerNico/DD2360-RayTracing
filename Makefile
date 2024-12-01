CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Debug flags including line info for profiling
NVCC_DBG       = -g -G -Xcompiler -rdynamic -src-in-ptx 

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_75,code=sm_75 # T4 GPU (Google Colab)

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile: cudart
	nsys profile \
		--stats=true \
		--backtrace=dwarf \
		--trace=cuda,nvtx \
		--cuda-memory-usage=true \  # Track memory allocations
		--force-overwrite=true \    # Overwrite existing reports
		--sample=cpu \              # CPU sampling
		--wait=all \                # Wait for all processes
		./cudart 1

clean:
	rm -f cudart cudart.o out.ppm out.jpg *.nsys-rep *.sqlite