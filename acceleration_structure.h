#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <queue>
#include <cuda_runtime.h>
#include "sphere.h"

struct SphereData
{
    float cx, cy, cz; // center
    float radius;
};

// Axis-aligned bounding box (AABB)
struct AABB {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;

    __host__ __device__ AABB()
        : minX(std::numeric_limits<float>::max()),
        minY(std::numeric_limits<float>::max()),
        minZ(std::numeric_limits<float>::max()),
        maxX(std::numeric_limits<float>::lowest()),
        maxY(std::numeric_limits<float>::lowest()),
        maxZ(std::numeric_limits<float>::lowest()) {
    }

    // Expand this AABB to include a sphere
    void expand(const sphere& s) {
        float cx = s.center.x();
        float cy = s.center.y();
        float cz = s.center.z();
        float r = s.radius;
        minX = std::min(minX, cx - r);
        minY = std::min(minY, cy - r);
        minZ = std::min(minZ, cz - r);
        maxX = std::max(maxX, cx + r);
        maxY = std::max(maxY, cy + r);
        maxZ = std::max(maxZ, cz + r);
    }
};

// Octree node that stores spheres
class OctreeNode {
public:
    AABB bounds;
    std::vector<int> objectIndices;
    std::unique_ptr<OctreeNode> children[8]; // 8 octants

    // Insert a single sphere index
    void insert(int sphereIndex, const std::vector<sphere>& allSpheres, int maxObjectsPerNode, int maxDepth, int currentDepth = 0)
    {
        // Expand bounding box to include this sphere
        bounds.expand(allSpheres[sphereIndex]);

        // If less than max or we've hit max depth, store in this node
        if ((int)objectIndices.size() < maxObjectsPerNode || currentDepth >= maxDepth) {
            objectIndices.push_back(sphereIndex);
            return;
        }

        // Otherwise subdivide if we haven't yet
        if (!children[0]) {
            subdivide();
        }

        // Attempt to place sphere into exactly one child if it fits fully
        for (int i = 0; i < 8; i++) {
            if (childContainsSphere(children[i]->bounds, allSpheres[sphereIndex])) {
                children[i]->insert(sphereIndex, allSpheres, maxObjectsPerNode, maxDepth, currentDepth + 1);
                return;
            }
        }

        // If sphere doesn't fully fit into any single child, keep it here
        objectIndices.push_back(sphereIndex);
    }

private:
    void subdivide()
    {
        float midX = 0.5f * (bounds.minX + bounds.maxX);
        float midY = 0.5f * (bounds.minY + bounds.maxY);
        float midZ = 0.5f * (bounds.minZ + bounds.maxZ);

        for (int i = 0; i < 8; i++) {
            children[i] = std::make_unique<OctreeNode>();
            children[i]->bounds.minX = (i & 1) ? midX : bounds.minX;
            children[i]->bounds.maxX = (i & 1) ? bounds.maxX : midX;
            children[i]->bounds.minY = (i & 2) ? midY : bounds.minY;
            children[i]->bounds.maxY = (i & 2) ? bounds.maxY : midY;
            children[i]->bounds.minZ = (i & 4) ? midZ : bounds.minZ;
            children[i]->bounds.maxZ = (i & 4) ? bounds.maxZ : midZ;
        }
    }

    bool childContainsSphere(const AABB& childBox, const sphere& s) const
    {
        float cx = s.center.x();
        float cy = s.center.y();
        float cz = s.center.z();
        float r = s.radius;
        return (cx - r >= childBox.minX && cx + r <= childBox.maxX &&
            cy - r >= childBox.minY && cy + r <= childBox.maxY &&
            cz - r >= childBox.minZ && cz + r <= childBox.maxZ);
    }
};

// CPU Octree: builds a root OctreeNode, then can be flattened
class Octree {
public:
    Octree(int maxObjPerNode = 4, int maxD = 8)
        : maxObjectsPerNode(maxObjPerNode), maxDepth(maxD) {
        root = std::make_unique<OctreeNode>();
    }

    void build(const std::vector<sphere>& spheres)
    {
        for (int i = 0; i < (int)spheres.size(); i++) {
            root->insert(i, spheres, maxObjectsPerNode, maxDepth, 0);
        }
    }

    // Flatten to arrays so we can upload to GPU
    // We'll do a breadth-first traversal to store each node in an array.
    // Children are stored by index in the same array; -1 means no child.
    struct FlatNode {
        // bounds
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
        // indices of children
        int children[8];
        // object list start and count
        int firstObj;
        int objCount;
    };

    // The result of flattening everything
    std::vector<FlatNode> flatNodes;
    std::vector<int> flatIndices; // indices of spheres

    void flatten()
    {
        flatNodes.clear();
        flatIndices.clear();
        // BFS
        std::queue<OctreeNode*> Q;
        std::queue<int> qIndex;  // index in flatNodes

        flatNodes.reserve(1024); // arbitrary
        rootIdx = 0;
        flatNodes.push_back(makeFlatNode(*root));
        Q.push(root.get());
        qIndex.push(0);

        while (!Q.empty()) {
            OctreeNode* node = Q.front(); Q.pop();
            int idx = qIndex.front();     qIndex.pop();

            // children
            for (int c = 0; c < 8; c++) {
                if (node->children[c]) {
                    // create new FlatNode in flatNodes
                    FlatNode fn = makeFlatNode(*node->children[c]);
                    int newIdx = (int)flatNodes.size();
                    flatNodes.push_back(fn);
                    // store index
                    flatNodes[idx].children[c] = newIdx;
                    // push child in queue
                    Q.push(node->children[c].get());
                    qIndex.push(newIdx);
                }
                else {
                    flatNodes[idx].children[c] = -1;
                }
            }
            // objectIndices
            // Because we store them contiguously in flatIndices,
            // we record their offset in 'firstObj' and size in 'objCount'
            flatNodes[idx].firstObj = (int)flatIndices.size();
            flatNodes[idx].objCount = (int)node->objectIndices.size();
            for (int sIdx : node->objectIndices) {
                flatIndices.push_back(sIdx);
            }
        }
    }

private:
    std::unique_ptr<OctreeNode> root;
    int rootIdx;
    int maxObjectsPerNode;
    int maxDepth;

    FlatNode makeFlatNode(const OctreeNode& n)
    {
        FlatNode fn;
        fn.minX = n.bounds.minX; fn.minY = n.bounds.minY; fn.minZ = n.bounds.minZ;
        fn.maxX = n.bounds.maxX; fn.maxY = n.bounds.maxY; fn.maxZ = n.bounds.maxZ;
        for (int i = 0; i < 8; i++) fn.children[i] = -1; // default
        fn.firstObj = 0;
        fn.objCount = 0;
        return fn;
    }
};

// GPU data for the octree
struct GpuOctree {
    Octree::FlatNode* nodes = nullptr;   // array of FlatNodes
    int* indices = nullptr; // array of sphere indices
    int               numNodes = 0;
    int               numIndices = 0;
};

// Copy the flattened data to device memory
inline void uploadOctreeToGPU(const std::vector<Octree::FlatNode>& fnodes,
    const std::vector<int>& findices,
    GpuOctree& outGpu)
{
    outGpu.numNodes = (int)fnodes.size();
    outGpu.numIndices = (int)findices.size();

    if (outGpu.numNodes == 0) return; // no data

    cudaMalloc((void**)&outGpu.nodes, outGpu.numNodes * sizeof(Octree::FlatNode));
    cudaMalloc((void**)&outGpu.indices, outGpu.numIndices * sizeof(int));

    cudaMemcpy(outGpu.nodes, fnodes.data(), outGpu.numNodes * sizeof(Octree::FlatNode), cudaMemcpyHostToDevice);
    cudaMemcpy(outGpu.indices, findices.data(), outGpu.numIndices * sizeof(int), cudaMemcpyHostToDevice);
}

// Free the GPU data
inline void freeGpuOctree(GpuOctree& oct)
{
    if (oct.nodes)   cudaFree(oct.nodes);
    if (oct.indices) cudaFree(oct.indices);
    oct.nodes = nullptr;
    oct.indices = nullptr;
    oct.numNodes = 0;
    oct.numIndices = 0;
}

// Device function for ray-box intersection test
__device__ bool intersectAABB(const float3& rayOrig,
    const float3& invDir,
    const AABB& box)
{
    // Slab test
    float t1 = (box.minX - rayOrig.x) * invDir.x;
    float t2 = (box.maxX - rayOrig.x) * invDir.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    t1 = (box.minY - rayOrig.y) * invDir.y;
    t2 = (box.maxY - rayOrig.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));

    t1 = (box.minZ - rayOrig.z) * invDir.z;
    t2 = (box.maxZ - rayOrig.z) * invDir.z;
    tmin = fmaxf(tmin, fminf(t1, t2));
    tmax = fminf(tmax, fmaxf(t1, t2));

    return (tmax > fmaxf(tmin, 0.0f));
}

// Device sphere intersection
__device__ bool intersectSphere(const float3& rayOrig,
    const float3& rayDir,
    const SphereData& s,
    float& outT)
{
    float ox = rayOrig.x - s.cx;
    float oy = rayOrig.y - s.cy;
    float oz = rayOrig.z - s.cz;
    float dx = rayDir.x;
    float dy = rayDir.y;
    float dz = rayDir.z;
    float r = s.radius;

    float A = dx * dx + dy * dy + dz * dz;
    float B = 2.f * (ox * dx + oy * dy + oz * dz);
    float C = ox * ox + oy * oy + oz * oz - r * r;
    float disc = B * B - 4.f * A * C;
    if (disc < 0.f) return false;

    float sqrtDisc = sqrtf(disc);
    float t1 = (-B - sqrtDisc) / (2.f * A);
    float t2 = (-B + sqrtDisc) / (2.f * A);

    float tHit = -1.f;
    if (t1 > 0.f && t2 > 0.f) tHit = fminf(t1, t2);
    else if (t1 > 0.f)      tHit = t1;
    else if (t2 > 0.f)      tHit = t2;
    if (tHit < 0.f) return false; // behind origin
    outT = tHit;
    return true;
}

// Device function: octree traversal to find the first sphere intersection
//  - stack-based BFS approach in device
__device__ float octreeIntersectRay(const float3& rayOrig,
    const float3& rayDir,
    const GpuOctree& oct,
    const SphereData* gpuSpheres,
    int numSpheres)
{
    if (oct.numNodes == 0) return -1.f; // no data

    // Precompute inverse dir (avoid zero)
    float3 invDir = make_float3((rayDir.x == 0.f) ? 1e32f : 1.f / rayDir.x,
        (rayDir.y == 0.f) ? 1e32f : 1.f / rayDir.y,
        (rayDir.z == 0.f) ? 1e32f : 1.f / rayDir.z);

    float closestT = 1e30f; // track nearest intersection
    // stack
    const int maxStack = 64;
    int stackArr[maxStack];
    int stackTop = 0;
    stackArr[stackTop++] = 0; // root is node 0

    while (stackTop > 0) {
        int nodeId = stackArr[--stackTop];
        // read node
        const Octree::FlatNode& nd = oct.nodes[nodeId];
        // bounds
        AABB box;
        box.minX = nd.minX; box.minY = nd.minY; box.minZ = nd.minZ;
        box.maxX = nd.maxX; box.maxY = nd.maxY; box.maxZ = nd.maxZ;

        // box check
        if (!intersectAABB(rayOrig, invDir, box)) continue;

        // check local objects
        for (int i = 0; i < nd.objCount; i++) {
            int sIdx = oct.indices[nd.firstObj + i];
            if (sIdx < 0 || sIdx >= numSpheres) continue;
            float tHit = 1e30f;
            if (intersectSphere(rayOrig, rayDir, gpuSpheres[sIdx], tHit)) {
                if (tHit < closestT) {
                    closestT = tHit;
                }
            }
        }
        // push children
        for (int c = 0; c < 8; c++) {
            int childId = nd.children[c];
            if (childId >= 0) {
                stackArr[stackTop++] = childId;
            }
        }
    }
    return (closestT < 1e30f) ? closestT : -1.f;
}
