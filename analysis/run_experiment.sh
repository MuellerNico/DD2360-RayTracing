#!/bin/bash

# List of executable paths
executables=(RayTracing_BASELINE_1000_01 RayTracing_OCTREE_1000_01
RayTracing_BASELINE_1000_02 RayTracing_OCTREE_1000_02
RayTracing_BASELINE_2000_01 RayTracing_OCTREE_2000_01
RayTracing_BASELINE_2000_02 RayTracing_OCTREE_2000_02
RayTracing_BASELINE_3000_01 RayTracing_OCTREE_3000_01
RayTracing_BASELINE_3000_02 RayTracing_OCTREE_3000_02
RayTracing_BASELINE_4000_01 RayTracing_OCTREE_4000_01
RayTracing_BASELINE_4000_02 RayTracing_OCTREE_4000_02
RayTracing_BASELINE_488_01 RayTracing_OCTREE_488_01
RayTracing_BASELINE_488_02 RayTracing_OCTREE_488_02
RayTracing_BASELINE_5000_01 RayTracing_OCTREE_5000_01
RayTracing_BASELINE_5000_02 RayTracing_OCTREE_5000_02
RayTracing_BASELINE_6000_01 RayTracing_OCTREE_6000_01
RayTracing_BASELINE_6000_02 RayTracing_OCTREE_6000_02
RayTracing_BASELINE_7000_01 RayTracing_OCTREE_7000_01
RayTracing_BASELINE_7000_02 RayTracing_OCTREE_7000_02
RayTracing_BASELINE_8000_01 RayTracing_OCTREE_8000_01
RayTracing_BASELINE_8000_02 RayTracing_OCTREE_8000_02
RayTracing_BASELINE_9000_01 RayTracing_OCTREE_9000_01
RayTracing_BASELINE_9000_02 RayTracing_OCTREE_9000_02)

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
