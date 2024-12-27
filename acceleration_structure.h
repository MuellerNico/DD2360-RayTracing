
#include "sphere.h"

#define CHILDREN_COUNT 8 // should not be changed, because it's an octree!
#define TREE_HEIGHT 3
#define NUMBER_NODES (1 + CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT + CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define NUMBER_LEAFS (CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT * CHILDREN_COUNT) // Tree height == 3
#define SPHERES_PER_LEAF 5

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
	*root = { 0, -11, -10000, -11, 11, 1, 11 };	// assuming whole world is in coordinate range [-50, 50] in all directions
	octree->nodeCount++;

	for (int i = 0; i < num_hitables; i++) {
		sphere curr_sphere = d_list[i];

		// insert recursively into tree
		int duplicateCount = insert(octree, root, curr_sphere, i);
		std::cout << "sphere " << i << " duplicated " << duplicateCount << " times in tree\n";
	}

	return octree;
}