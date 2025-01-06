#!/bin/bash

# List of executable paths
executables=(
    RayTracing_MAX_OCTREE_1093_01 RayTracingBaseline RayTracingFP16 RayTracing_MAX_OCTREE_1093_02
    RayTracing_MAX_OCTREE_1940_01 RayTracing_MAX_OCTREE_1940_02 RayTracing_MAX_OCTREE_3029_01
    RayTracing_MAX_OCTREE_3029_02 RayTracing_MAX_OCTREE_4360_01 RayTracing_MAX_OCTREE_4360_02
    RayTracing_MAX_OCTREE_488_01 RayTracing_MAX_OCTREE_488_02 RayTracing_MAX_OCTREE_7748_01
    RayTracing_MAX_OCTREE_7748_02 RayTracing_MAX_BASELINE_1093_01 RayTracing_MAX_BASELINE_1093_02
    RayTracing_MAX_BASELINE_1940_01 RayTracing_MAX_BASELINE_1940_02 RayTracing_MAX_BASELINE_3029_01
    RayTracing_MAX_BASELINE_3029_02 RayTracing_MAX_BASELINE_4360_01 RayTracing_MAX_BASELINE_4360_02
    RayTracing_MAX_BASELINE_488_01 RayTracing_MAX_BASELINE_488_02 RayTracing_MAX_BASELINE_7748_01
    RayTracing_MAX_BASELINE_7748_02
)

# Loop over each executable
for executable in "${executables[@]}"; do
    # Create folder for each executable
    mkdir -p ./experiments/${executable}

    for i in {1..5}; do
        echo "Running iteration $i for experiment: ${executable}"

        # Run ncu in parallel
        ncu --section LaunchStats --section SpeedOfLight --section MemoryWorkloadAnalysis \
            --section ComputeWorkloadAnalysis --section Occupancy --export ./experiments/${executable}/${executable}_${i} --csv ./$executable 3 &

        ncu_pid=$!

        # Monitor GPU memory usage every 200ms until ncu completes
        while kill -0 $ncu_pid 2>/dev/null; do
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> ./experiments/${executable}/${executable}_${i}_gpu_memory.csv
            sleep 0.2
        done

        # Rename the output image
        if [ -f output.ppm ]; then
            mv output.ppm ./experiments/${executable}/${executable}_${i}.ppm
        else
            echo "Warning: output.ppm not found for iteration $i"
        fi

        # Run the second command to generate CSV
        ncu --csv --import ./experiments/${executable}/${executable}_${i}.ncu-rep > ./experiments/${executable}/${executable}_${i}.csv

    done

done

echo "All experiments completed successfully."
