/**
 * @file acceleration_structure.h
 * @brief Implementation of an octree-based acceleration structure for ray tracing
 * 
 * This file contains the implementation of an octree spatial partitioning structure
 * used to accelerate ray-sphere intersection tests in a CUDA-based ray tracer.
 */

#include "sphere.h"

#define CHILDREN_COUNT 8 // should not be changed, because it's an octree!
#define TREE_HEIGHT 3
#define NUMBER_NODES (1 + CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define NUMBER_LEAFS (CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define SPHERES_PER_LEAF 30 // needs to be adjusted when NUM_SPHERES is changed

/**
 * @brief Axis-Aligned Bounding Box (AABB) structure
 * 
 * Represents a 3D box aligned with the coordinate axes, defined by its minimum
 * and maximum coordinates in each dimension.
 */
struct AABB {
	real_t x_low, y_low, z_low;
	real_t x_high, y_high, z_high;
};

/**
 * @brief Node structure for the octree
 * 
 * Represents an internal node in the octree. Leaf nodes are handled separately
 * in the OctLeaf structure.
 */
struct OctNode {
    int level;                      ///< Current depth in the tree (3 indicates leaf parent)
    AABB aabb;                      ///< Bounding box for this node
    int children[CHILDREN_COUNT];    ///< Indices of child nodes or leaves
};

/**
 * @brief Leaf node structure for the octree
 * 
 * Stores references to the spheres that intersect with this leaf's volume.
 */
struct OctLeaf {
    int sphere_indices[SPHERES_PER_LEAF];  ///< Indices of spheres in this leaf
    int index_count;                       ///< Number of spheres currently stored
};

/**
 * @brief Main octree structure
 * 
 * Contains all nodes and leaves of the octree, along with counters for
 * tracking the number of allocated nodes and leaves.
 */
struct Octree {
    OctNode nodes[NUMBER_NODES];      ///< Array of all nodes in the tree
    OctLeaf leaves[NUMBER_LEAFS + 1]; ///< Array of all leaves (index 0 is skipped)
    int nodeCount = 0;                ///< Number of nodes currently in use
    int leafCount = 1;                ///< Number of leaves currently in use
};

/**
 * @brief Structure to track intersection testing results
 * 
 * Used during ray traversal to keep track of the closest intersection
 * and associated data.
 */
struct ProcessedHit {
    bool hit_anything;
    hit_record rec;
    real_t closest_so_far;
};

/**
 * @brief Tests if a sphere intersects with an AABB
 * 
 * @param obj The sphere to test
 * @param aabb The axis-aligned bounding box
 * @return true if the sphere intersects the AABB, false otherwise
 */
inline bool intersects(const sphere& obj, AABB aabb) {
	aabb.x_low -= obj.radius;
	aabb.y_low -= obj.radius;
	aabb.z_low -= obj.radius;
	aabb.x_high += obj.radius;
	aabb.y_high += obj.radius;
	aabb.z_high += obj.radius;

	return (obj.center.x() > aabb.x_low && obj.center.x() <= aabb.x_high)
		&& (obj.center.y() >= aabb.y_low && obj.center.y() <= aabb.y_high)
		&& (obj.center.z() >= aabb.z_low && obj.center.z() <= aabb.z_high);
}

/**
 * @brief Inserts a sphere into the octree
 * 
 * @param octree Pointer to the octree
 * @param node Current node being processed
 * @param obj Sphere to insert
 * @param sphereidx Index of the sphere in the global sphere array
 * @return Number of leaf nodes the sphere was inserted into
 */
inline int insert(Octree* octree, OctNode* node, const sphere& obj, int sphereidx) {
	if (!intersects(obj, node->aabb)) {
		printf("Something went wrong. Why is sphere not in range of nodes AABB? Sphere origin: %f %f %f\n", obj.center.e[0], obj.center.e[1], obj.center.e[2]);
		return 0;
	}

	if (node->level == TREE_HEIGHT) {
		// found last node before leaf; insert
		for (int i = 0; i < 8; i++) {
			int leaf_index = node->children[i];
			if(leaf_index == 0)
			{
				// create leaf
				leaf_index = octree->leafCount++;
				octree->leaves[leaf_index] = {}; // zero-init
				node->children[i] = leaf_index;
			}

			OctLeaf* curr_leaf = &octree->leaves[leaf_index];

			if(curr_leaf->index_count < SPHERES_PER_LEAF)
			{
				// enough space, insert sphere
				curr_leaf->sphere_indices[curr_leaf->index_count++] = sphereidx;
				return 1;
			} else
			{
				// need to insert in next leaf
				continue;
			}
		}
		printf("Something went wrong. Leaf nodes full. Extends: %f %f %f x %f %f %f\n", node->aabb.x_low, node->aabb.y_low, node->aabb.z_low, node->aabb.x_high, node->aabb.y_high, node->aabb.z_high);
		return 0;
	}

	int insertCounter = 0;

	auto halfs = [](float low, float high) -> float { return  (low + (high - low) / 2); };

	auto [x_low, y_low, z_low, x_high, y_high, z_high] = node->aabb;

	float x_half = halfs(x_low, x_high);
	float y_half = halfs(y_low, y_high);
	float z_half = halfs(z_low, z_high);

	// binary low-low-low, low-low-high, low-high-low, ...
	AABB childrenBoxes[8] = {	{x_low, y_low, z_low,
								x_half, y_half, z_half },
								{ x_low, y_low, z_half,
								x_half, y_half, z_high },
								{ x_low, y_half, z_low,
								x_half, y_high, z_half },
								{ x_low, y_half, z_half,
								x_half, y_high, z_high },
								{ x_half, y_low, z_low,
								x_high, y_half, z_half },
								{ x_half, y_low, z_half,
								x_high, y_half, z_high },
								{ x_half, y_half, z_low,
								x_high, y_high, z_half },
								{ x_half, y_half, z_half,
								x_high, y_high, z_high } };

	for (int i = 0; i < 8; i++) {
		if (intersects(obj, childrenBoxes[i])) {
			// -> insert in child node
			if (node->children[i] == 0) {
				// child doesn't exist -> create child node 
				// (index 0 is only root, root will never be a child)
				node->children[i] = octree->nodeCount;
				octree->nodes[octree->nodeCount++] = { node->level + 1, childrenBoxes[i], { 0 } };

				insertCounter += insert(octree, &octree->nodes[node->children[i]], obj, sphereidx);
			}
			else {
				// insert into existing node
				insertCounter += insert(octree, &octree->nodes[node->children[i]], obj, sphereidx);
			}
		}
	}

	return insertCounter; // keep track of duplicate insertion
}

/**
 * @brief Builds an octree from an array of spheres
 * 
 * @param d_list Array of spheres to include in the tree
 * @param num_hitables Number of spheres in the array
 * @return Pointer to the constructed octree
 */
Octree* buildOctree(sphere* d_list, const int num_hitables) {

	Octree* octree = new Octree();

	// initialize root node
	OctNode* root = &(octree->nodes[0]);
	*root = {
		0,
		{ -11, 0, -11, 11, 2, 11 },  // AABB
		{ 0,0,0,0,0,0,0,0 }          // children
	};
	octree->nodeCount++;

	for (int i = 1; i < num_hitables; i++) { // skip ground sphere (idx 0)
		sphere curr_sphere = d_list[i];

		// insert recursively into tree
		int duplicateCount = insert(octree, root, curr_sphere, i);
		//std::cout << "sphere " << i << " duplicated " << duplicateCount << " times in tree\n"; // debug
	}

	return octree;
}

/**
 * @brief Tests if a ray intersects with an AABB
 * 
 * @param r The ray to test
 * @param box The axis-aligned bounding box
 * @return true if the ray intersects the AABB, false otherwise
 */
__device__ bool intersect_ray_aabb(const ray& r, const AABB& box) {
	float tmin = (box.x_low - r.origin().x()) / r.direction().x();
	float tmax = (box.x_high - r.origin().x()) / r.direction().x();
	if (tmin > tmax) { float tmp = tmin; tmin = tmax; tmax = tmp; }

	float tymin = (box.y_low - r.origin().y()) / r.direction().y();
	float tymax = (box.y_high - r.origin().y()) / r.direction().y();
	if (tymin > tymax) { float tmp = tymin; tymin = tymax; tymax = tmp; }
	if ((tmin > tymax) || (tymin > tmax)) return false;
	if (tymin > tmin) tmin = tymin;
	if (tymax < tmax) tmax = tymax;

	float tzmin = (box.z_low - r.origin().z()) / r.direction().z();
	float tzmax = (box.z_high - r.origin().z()) / r.direction().z();
	if (tzmin > tzmax) { float tmp = tzmin; tzmin = tzmax; tzmax = tmp; }
	if ((tmin > tzmax) || (tzmin > tmax)) return false;

	return true;
}

/**
 * @brief Processes a potential hit between a ray and a sphere
 * 
 * @param r The ray being traced
 * @param sphere_idx Index of the sphere to test
 * @param result Structure to store the intersection result
 * @param world Pointer to the world containing all objects
 */
__device__ void processHit(const ray& r, const int sphere_idx, ProcessedHit& result, hitable** world) {
    if (sphere_idx == 0) return; // should never happen, but just in case. Ground sphere is not in tree
    hit_record temp_rec;
    hitable* hit_pointer = ((hitable_list*)(*world))->list[sphere_idx];
    sphere sphere_obj = *((sphere*)hit_pointer);
    
    if (sphere_obj.hit(r, real_t(0.001f), result.closest_so_far, temp_rec)) {
        result.hit_anything = true;
        result.closest_so_far = temp_rec.t;
        result.rec = temp_rec;
    }
}

/**
 * @brief Recursively traverses the octree to find ray intersections
 * 
 * @param octree Pointer to the octree
 * @param r The ray being traced
 * @param curr_node Current node being processed
 * @param result Structure to store the intersection result
 * @param world Pointer to the world containing all objects
 */
__device__ void traverseTree(Octree* octree, const ray& r, OctNode* curr_node, ProcessedHit& result, hitable** world) {
    if(!intersect_ray_aabb(r, curr_node->aabb)) {
        return; // no intersection possible in this node
    }

    if (curr_node->level == TREE_HEIGHT) { // is parent of leaf
        for (int i = 0; i < 8; i++) {
            int leaf_index = curr_node->children[i];
            if(leaf_index == 0) { // no leaf at idx 0
                return;
            } else if (leaf_index >= octree->leafCount) {
                printf("leaf_index should never be bigger than leaf_count. Something went wrong.\n");
                return;
            }
			// iterate over spheres in leaf and check hit
            OctLeaf& curr_leaf = octree->leaves[leaf_index];
            for(int j=0; j < curr_leaf.index_count; j++) { 
                processHit(r, curr_leaf.sphere_indices[j], result, world); 
            }
        }
        return;
    }

	// recursively traverse children
    for(int i = 0; i < CHILDREN_COUNT; i++) {
        if(curr_node->children[i] != 0)
            traverseTree(octree, r, &octree->nodes[curr_node->children[i]], result, world);
    }
}

/**
 * @brief Tests if a ray intersects with any object in the octree
 * 
 * This is the main entry point for ray intersection testing using the octree.
 * It first checks against the ground sphere (index 0) and then traverses the
 * octree to find the closest intersection with any other sphere.
 * 
 * @param octree Pointer to the octree
 * @param r The ray to test
 * @param rec Record to store intersection data
 * @param world Pointer to the world containing all objects
 * @return true if the ray intersects any object, false otherwise
 */
__device__ bool hitTree(Octree* octree, const ray& r, hit_record& rec, hitable** world) {
    
    // first check ground sphere (index 0)
    hit_record ground_rec;
    hitable* hit_pointer = ((hitable_list*)(*world))->list[0];
    sphere sphere_obj = *((sphere*)hit_pointer);
    bool hit_ground = sphere_obj.hit(r, 0.001f, FLT_MAX, ground_rec);
    
    // init result with ground sphere's hit distance if it was hit
    ProcessedHit result = {false, {}, hit_ground ? ground_rec.t : real_t(FLT_MAX)};
    if (hit_ground) {
        result.hit_anything = true;
        result.rec = ground_rec;
    }
        
    traverseTree(octree, r, &octree->nodes[0], result, world);
    
    // Return final result
    if (result.hit_anything) {
        rec = result.rec;
        return true;
    }
    return false;
}
