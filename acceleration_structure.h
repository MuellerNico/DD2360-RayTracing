
#include "sphere.h"

struct AABB {
	float x_low, y_low, z_low;
	float x_high, y_high, z_high;
};

struct OctNode {	// is leaf at last level
	int level; // == 3 indicates leaf
	AABB aabb;
	int children[8];
};

struct Octree {
	OctNode nodes[8 * 8 * 8 * 8];	// max 4 levels
	int nodeCount = 0;
};

bool intersects(const sphere& obj, AABB aabb) {
	aabb.x_low -= obj.radius;
	aabb.y_low -= obj.radius;
	aabb.z_low -= obj.radius;
	aabb.x_high += obj.radius;
	aabb.y_high += obj.radius;
	aabb.z_high += obj.radius;

	return (obj.center.x >= aabb.x_low && obj.center.x <= aabb.x_high)
		&& (obj.center.y >= aabb.y_low && obj.center.y <= aabb.y_high)
		&& (obj.center.z >= aabb.z_low && obj.center.z <= aabb.z_high)
}

int insert(Octree octree, OctNode* node, const sphere& obj, int sphereidx) {
	if (!intersects(obj, node.aabb)) {
		printf("Something went wrong. Why is sphere not in range of nodes AABB?");
		return 0;
	}

	if (node->level == 3) {
		// found leaf; insert
		for (int i = 0; i < 8; i++) {
			if (node->children[i] == -1) {
				// empty spot -> insert
				node->children[i] = sphereidx;
				return 1;
			}
		}
		printf("Something went wrong. Leaf node full.");
		return 0;
	}

	int insertCounter = 0;

	float halfs(float low, float high) [] -> { return  (low + (high - low)); }

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
								{ x_low, y_half z_half,
								x_half, y_high, z_high },
								{ x_half, y_low, z_low,
								x_high, y_half, z_half },
								{ x_half, y_low, z_half,
								x_high, y_half, z_high },
								{ x_half, y_half, z_low,
								x_high, y_high, z_half },
								{ x_half, y_half z_half,
								x_high, y_high, z_high } };

	for (int i = 0; i < 8; i++) {
		if (intersects(sphere, childrenBoxes[i])) {
			// -> insert in child node
			if (node->children[i] == 0) {
				// child doesn't exist -> create child node 
				// (index 0 is only root, root will never be a child)
				node->children[i] = octree.nodeCount;
				octree.nodes[octree.nodeCount++] = { node->level + 1, childrenBoxes[i], { 0 } };

				insertCounter += insert(octree, octree.nodes[octree.nodeCount - 1], obj, sphereidx);
			}
			else {
				// insert into existing node
				insertCounter += insert(octree, octree.nodes[node->children[i]], obj, sphereidx);
			}
		}
	}

	return insertCounter;
}

Octree* buildOctree(hitable** d_list, const int num_hitables) {

	Octree* octree = new Octree();
	OctNode* root = &octree->nodes[0];
	octree->nodeCount++;

	// initialize root node
	*root = { 0, -5, -5, -5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1 };	// assuming whole world is in coordinate range [-5, 5] in all directions

	for (int i = 0; i < num_hitables; i++) {
		sphere* curr_sphere = d_list[i];	// assuming its all spheres

		// insert recursively into tree
		int duplicateCount = insert(octree, root, *curr_sphere, i);
	}
}