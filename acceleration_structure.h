
#include "sphere.h"

#define CHILDREN_COUNT 8 // should not be changed, because it's an octree!
#define TREE_HEIGHT 3
#define NUMBER_NODES (1 + CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define NUMBER_LEAFS (CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define SPHERES_PER_LEAF 5
#define NUM_SPHERES (22 * 22 + 1 + 3)

struct AABB {
	float x_low, y_low, z_low;
	float x_high, y_high, z_high;
};

struct OctNode {	// is leaf at last level
	int level; // == 3 indicates leaf
	AABB aabb;
	int children[CHILDREN_COUNT];
};

struct OctLeaf
{
	int sphere_indices[SPHERES_PER_LEAF];
	int index_count;
};

struct Octree {
	OctNode nodes[NUMBER_NODES];
	OctLeaf leaves[NUMBER_LEAFS + 1]; // skip index 0 
	int nodeCount = 0;
	int leafCount = 1;	// workaround
};

struct Octhit {
	static const int MAX_HITS = 600;
	int possible_hits[MAX_HITS];
	int num_p_hits;
};

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
			}else
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

	return insertCounter;
}

Octree* buildOctree(sphere* d_list, const int num_hitables) {

	Octree* octree = new Octree();

	// initialize root node
	OctNode* root = &(octree->nodes[0]);
	*root = {
		0,
		{ -11, -10000, -11, 11, 1, 11 },  // AABB
		{ 0,0,0,0,0,0,0,0 }              // children
	};
	octree->nodeCount++;

	for (int i = 0; i < num_hitables; i++) {
		sphere curr_sphere = d_list[i];

		// insert recursively into tree
		int duplicateCount = insert(octree, root, curr_sphere, i);
		//std::cout << "sphere " << i << " duplicated " << duplicateCount << " times in tree\n";
	}

	return octree;
}

__device__ bool intersect_ray_aabb(const ray& r, const AABB& box) {
	// Handle cases where ray direction components are near zero
	const float epsilon = 1e-6f;

	float dirX = fabsf(r.direction().x()) < epsilon ? epsilon : r.direction().x();
	float dirY = fabsf(r.direction().y()) < epsilon ? epsilon : r.direction().y();
	float dirZ = fabsf(r.direction().z()) < epsilon ? epsilon : r.direction().z();

	float tmin = (box.x_low - r.origin().x()) / dirX;
	float tmax = (box.x_high - r.origin().x()) / dirX;
	if (tmin > tmax) { float tmp = tmin; tmin = tmax; tmax = tmp; }

	float tymin = (box.y_low - r.origin().y()) / dirY;
	float tymax = (box.y_high - r.origin().y()) / dirY;
	if (tymin > tymax) { float tmp = tymin; tymin = tymax; tymax = tmp; }

	if ((tmin > tymax) || (tymin > tmax)) return false;
	if (tymin > tmin) tmin = tymin;
	if (tymax < tmax) tmax = tymax;

	float tzmin = (box.z_low - r.origin().z()) / dirZ;
	float tzmax = (box.z_high - r.origin().z()) / dirZ;
	if (tzmin > tzmax) { float tmp = tzmin; tzmin = tzmax; tzmax = tmp; }

	if ((tmin > tzmax) || (tzmin > tmax)) return false;
	if (tzmin > tmin) tmin = tzmin;
	if (tzmax < tmax) tmax = tzmax;

	return tmax > 0; // Intersection is only valid if it's in front of the ray
}

/*
__device__ void traverseTree(Octree* octree, const ray& r, int nodeIndex, Octhit* hit)
{
	// TODO: find out why error is thrown in here
	// TODO: find out why ray seems to hit too much, resulting in Octhit returning more spheres than seem necessary
	OctNode& curr_node = octree->nodes[nodeIndex];
	
	if(!intersect_ray_aabb(r, curr_node.aabb))
	{
		// misses bb -> no further traversal
		return;
	}

	if (curr_node.level == TREE_HEIGHT) {
		// found last node before leaf; hits all children
		for (int i = 0; i < 8; i++) {
			int leaf_index = curr_node.children[i];
			if(leaf_index == 0)
			{
				// no leaf -> return
				continue;
			}else if (leaf_index >= octree->leafCount){
				printf("leaf_index should never be bigger than leaf_count. Something went wrong.\n");
				continue;
			}

			OctLeaf curr_leaf = octree->leaves[leaf_index];
			for(int j=0; j < curr_leaf.index_count && hit->num_p_hits < Octhit::MAX_HITS; j++)
			{
				int sphere_index = curr_leaf.sphere_indices[j];
				hit->possible_hits[hit->num_p_hits++] = sphere_index;	// add to hit result TODO error seems to be thrown here, but not sure
			}
		}
		return;
	}

	for(int i = 0; i < CHILDREN_COUNT; i++)
	{
		// check all existing children
		if(curr_node.children[i] != 0)
			traverseTree(octree, r, curr_node.children[i], hit);
	}
}
*/


//__device__ void traverseTree(Octree* octree, const ray& r, int nodeIndex, Octhit* hit)
//{
//	// Debug the node access
//	printf("Accessing node %d\n", nodeIndex);
//	if (nodeIndex >= NUMBER_NODES) {
//		printf("Error: Invalid node index %d\n", nodeIndex);
//		return;
//	}
//
//	OctNode& curr_node = octree->nodes[nodeIndex];
//	printf("Node level: %d\n", curr_node.level);
//
//	// Debug the ray-AABB intersection
//	if (!intersect_ray_aabb(r, curr_node.aabb)) {
//		printf("Ray missed node %d AABB\n", nodeIndex);
//		return;
//	}
//	printf("Ray hit node %d AABB\n", nodeIndex);
//
//	if (curr_node.level == TREE_HEIGHT) {
//		printf("At leaf level for node %d\n", nodeIndex);
//		for (int i = 0; i < 8; i++) {
//			int leaf_index = curr_node.children[i];
//			if (leaf_index == 0 || leaf_index >= octree->leafCount) {
//				continue;
//			}
//			printf("Processing leaf %d with %d spheres\n",
//				leaf_index, octree->leaves[leaf_index].index_count);
//
//			OctLeaf& curr_leaf = octree->leaves[leaf_index];
//			for (int j = 0; j < curr_leaf.index_count && hit->num_p_hits < Octhit::MAX_HITS; j++) {
//				hit->possible_hits[hit->num_p_hits++] = curr_leaf.sphere_indices[j];
//				printf("Added sphere %d to hits\n", curr_leaf.sphere_indices[j]);
//			}
//		}
//		return;
//	}
//
//	for (int i = 0; i < CHILDREN_COUNT; i++) {
//		if (curr_node.children[i] != 0) {
//			printf("Traversing from node %d to child %d\n", nodeIndex, curr_node.children[i]);
//			traverseTree(octree, r, curr_node.children[i], hit);
//		}
//	}
//}
//
//
//
//


//__device__ void traverseTree(Octree* octree, const ray& r, int nodeIndex, Octhit* hit) {
//	printf("Entering traverseTree for node %d\n", nodeIndex);
//	// First, validate the node index before any memory access
//	if (nodeIndex < 0 || nodeIndex >= NUMBER_NODES) {
//		printf("Error: Invalid node index %d\n", nodeIndex);
//		return;
//	}
//
//	// Early exit if hit buffer is full
//	if (hit->num_p_hits >= Octhit::MAX_HITS) {
//		return;
//	}
//
//	// Get reference to current node
//	const OctNode& curr_node = octree->nodes[nodeIndex];
//
//	// Check intersection before doing any more work
//	if (!intersect_ray_aabb(r, curr_node.aabb)) {
//		printf("Ray missed node %d AABB\n", nodeIndex);
//		return;
//	}
//
//	printf("Processing node %d (level %d)\n", nodeIndex, curr_node.level);
//
//	if (curr_node.level == TREE_HEIGHT) {
//		// At leaf level - process contained spheres
//		for (int i = 0; i < CHILDREN_COUNT; i++) {
//			int leaf_index = curr_node.children[i];
//
//			// Skip invalid leaves
//			if (leaf_index <= 0 || leaf_index >= octree->leafCount) {
//				continue;
//			}
//
//			const OctLeaf& curr_leaf = octree->leaves[leaf_index];
//
//			// Process spheres in this leaf
//			for (int j = 0; j < curr_leaf.index_count && j < SPHERES_PER_LEAF; j++) {
//				int sphere_index = curr_leaf.sphere_indices[j];
//
//				// Validate sphere index
//				if (sphere_index >= 0 && sphere_index < NUM_SPHERES) {
//					// Double-check buffer capacity before writing
//					if (hit->num_p_hits < Octhit::MAX_HITS) {
//						hit->possible_hits[hit->num_p_hits++] = sphere_index;
//						printf("Added sphere %d from leaf %d\n", sphere_index, leaf_index);
//					}
//					else {
//						return;  // Buffer full
//					}
//				}
//			}
//		}
//	}
//	else {
//		// Internal node - traverse children
//		for (int i = 0; i < CHILDREN_COUNT; i++) {
//			int child_index = curr_node.children[i];
//			if (child_index > 0 && child_index < octree->nodeCount) {
//				traverseTree(octree, r, child_index, hit);
//
//				// Early exit if buffer is full
//				if (hit->num_p_hits >= Octhit::MAX_HITS) {
//					return;
//				}
//			}
//		}
//	}
//}

///////////////////////////////////////////////////


// First kernel: Just validate the node and check intersection
__device__ bool validateAndIntersectNode(Octree* octree, const ray& r, int nodeIndex, int* level) {
	printf("Validation check for node %d\n", nodeIndex);

	// Basic index validation
	if (nodeIndex < 0 || nodeIndex >= NUMBER_NODES) {
		printf("Error: Invalid node index %d\n", nodeIndex);
		return false;
	}

	// Get node and check AABB intersection
	const OctNode& curr_node = octree->nodes[nodeIndex];
	*level = curr_node.level;

	bool intersects = intersect_ray_aabb(r, curr_node.aabb);
	printf("Node %d intersection result: %d\n", nodeIndex, intersects);

	return intersects;
}

// Second kernel: Process a leaf node
__device__ void processLeafNode(Octree* octree, int nodeIndex, Octhit* hit) {
	printf("Processing leaf node %d\n", nodeIndex);

	const OctNode& curr_node = octree->nodes[nodeIndex];

	for (int i = 0; i < CHILDREN_COUNT; i++) {
		int leaf_index = curr_node.children[i];

		if (leaf_index <= 0 || leaf_index >= octree->leafCount) {
			printf("Skipping invalid leaf index %d\n", leaf_index);
			continue;
		}

		const OctLeaf& curr_leaf = octree->leaves[leaf_index];
		printf("Processing leaf %d with %d spheres\n", leaf_index, curr_leaf.index_count);

		for (int j = 0; j < curr_leaf.index_count && j < SPHERES_PER_LEAF; j++) {
			int sphere_index = curr_leaf.sphere_indices[j];

			if (sphere_index >= 0 && sphere_index < NUM_SPHERES && hit->num_p_hits < Octhit::MAX_HITS) {
				hit->possible_hits[hit->num_p_hits++] = sphere_index;
				printf("Added sphere %d from leaf %d\n", sphere_index, leaf_index);
			}
		}
	}
}

// declare traverseTree function
__device__ void traverseTree(Octree* octree, const ray& r, int nodeIndex, Octhit* hit);

// Third kernel: Handle internal node traversal
__device__ void processInternalNode(Octree* octree, const ray& r, int nodeIndex, Octhit* hit) {
	printf("Processing internal node %d\n", nodeIndex);

	const OctNode& curr_node = octree->nodes[nodeIndex];

	for (int i = 0; i < CHILDREN_COUNT; i++) {
		int child_index = curr_node.children[i];
		if (child_index > 0 && child_index < octree->nodeCount) {
			printf("Traversing to child %d from node %d\n", child_index, nodeIndex);
			traverseTree(octree, r, child_index, hit);
		}
	}
}

// Main traversal function now uses these separate components
__device__ void traverseTree(Octree* octree, const ray& r, int nodeIndex, Octhit* hit) {
	printf("Starting traversal for node %d\n", nodeIndex);

	// Early exit if hit buffer is full
	if (hit->num_p_hits >= Octhit::MAX_HITS) {
		printf("Hit buffer full, early exit\n");
		return;
	}

	// Step 1: Validate and check intersection
	int node_level;
	if (!validateAndIntersectNode(octree, r, nodeIndex, &node_level)) {
		printf("Node %d validation or intersection failed\n", nodeIndex);
		return;
	}

	// Step 2: Process based on node type
	if (node_level == TREE_HEIGHT) {
		processLeafNode(octree, nodeIndex, hit);
	}
	else {
		processInternalNode(octree, r, nodeIndex, hit);
	}

	printf("Completed traversal for node %d\n", nodeIndex);
}


__device__ void hitTree(Octree* octree, const ray& r, Octhit& hit) {
	printf("=== Starting new ray traversal ===\n");

	if (octree == nullptr) {
		printf("ERROR: Null octree pointer\n");
		return;
	}

	hit.num_p_hits = 0;
	traverseTree(octree, r, 0, &hit);

	printf("=== Ray traversal complete, found %d hits ===\n", hit.num_p_hits);
}


//__device__ Octhit* hitTree(Octree* octree, const ray& r)
//{
//	Octhit* hit = new Octhit();
//	hit->possible_hits = new int[600];
//
//	traverseTree(octree, r, &octree->nodes[0], hit);	// start at root 0
//
//	return hit;
//}