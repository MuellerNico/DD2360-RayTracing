#pragma once
#include <vector>
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

    __host__ __device__
        AABB()
        : minX(std::numeric_limits<float>::max()),
        minY(std::numeric_limits<float>::max()),
        minZ(std::numeric_limits<float>::max()),
        maxX(std::numeric_limits<float>::lowest()),
        maxY(std::numeric_limits<float>::lowest()),
        maxZ(std::numeric_limits<float>::lowest()) {
    }

    __host__
        void expand(const sphere& s) {
	        const float cx = s.center.x();
	        const float cy = s.center.y();
	        const float cz = s.center.z();
	        const float r = s.radius;
        minX = std::min(minX, cx - r);
        minY = std::min(minY, cy - r);
        minZ = std::min(minZ, cz - r);
        maxX = std::max(maxX, cx + r);
        maxY = std::max(maxY, cy + r);
        maxZ = std::max(maxZ, cz + r);
    }
};

// FlatNode
//  - holds bounding box
//  - has indices to children
//  - has a list of sphere indices
struct FlatNode {
    // bounding box
    float minX, minY, minZ;
    float maxX, maxY, maxZ;

    // children indices (-1 means no child)
    int children[8];

    // range in a global "indices array" that lists which spheres are in this node
    int firstObj;
    int objCount;
};

// Holds arrays of FlatNode and sphere indices on the GPU
struct GpuOctree {
    FlatNode* nodes = nullptr;
    int* indices = nullptr;
    int       numNodes = 0;
    int       numIndices = 0;
};

// A small helper to free the GPU octree arrays
inline void freeGpuOctree(GpuOctree& oct)
{
    if (oct.nodes)   cudaFree(oct.nodes);
    if (oct.indices) cudaFree(oct.indices);
    oct.nodes = nullptr;
    oct.indices = nullptr;
    oct.numNodes = 0;
    oct.numIndices = 0;
}

// Minimal CPU-based structure to hold all flat nodes + the array of sphere indices
class Octree {
public:
    std::vector<FlatNode> flatNodes;
    std::vector<int>      flatIndices;

    Octree(const int maxObjPerNode = 4, const int maxD = 8)
        : maxObjectsPerNode(maxObjPerNode), maxDepth(maxD) {
    }

    // Top-level build: we create a single node (the root), then subdivide recursively
    void build(const std::vector<sphere>& spheres)
    {
        flatNodes.clear();
        flatIndices.clear();

        // We store all sphere indices in a temporary array
        std::vector<int> allIndices(spheres.size());
        for (int i = 0; i < static_cast<int>(spheres.size()); i++) {
            allIndices[i] = i;
        }

        // Build the root recursively
        buildNode(spheres, allIndices, /*depth=*/0, /*parentIdx=*/-1);
    }

private:
    int maxObjectsPerNode;
    int maxDepth;

    // Create a node & set default values
    // returns index of new node
    int createNode()
    {
        FlatNode fn;
        fn.minX = std::numeric_limits<float>::max();
        fn.minY = std::numeric_limits<float>::max();
        fn.minZ = std::numeric_limits<float>::max();
        fn.maxX = std::numeric_limits<float>::lowest();
        fn.maxY = std::numeric_limits<float>::lowest();
        fn.maxZ = std::numeric_limits<float>::lowest();
        for (int c = 0; c < 8; c++) fn.children[c] = -1;
        fn.firstObj = 0;
        fn.objCount = 0;

        flatNodes.push_back(fn);
        return static_cast<int>(flatNodes.size()) - 1;
    }

    // Recursively build a node from a set of sphere indices
    void buildNode(const std::vector<sphere>& spheres,
        const std::vector<int>& nodeIndices,
        const int depth,
        int parentIdx)
    {
        // create a new FlatNode
        const int thisNodeIdx = createNode();
        FlatNode& thisNode = flatNodes[thisNodeIdx];

        // expand bounding box
        for (const int idx : nodeIndices) {
            thisNode.minX = std::min(thisNode.minX, spheres[idx].center.x() - spheres[idx].radius);
            thisNode.minY = std::min(thisNode.minY, spheres[idx].center.y() - spheres[idx].radius);
            thisNode.minZ = std::min(thisNode.minZ, spheres[idx].center.z() - spheres[idx].radius);
            thisNode.maxX = std::max(thisNode.maxX, spheres[idx].center.x() + spheres[idx].radius);
            thisNode.maxY = std::max(thisNode.maxY, spheres[idx].center.y() + spheres[idx].radius);
            thisNode.maxZ = std::max(thisNode.maxZ, spheres[idx].center.z() + spheres[idx].radius);
        }

        // If we are under capacity or at maxDepth => store indices
        if (static_cast<int>(nodeIndices.size()) <= maxObjectsPerNode || depth >= maxDepth) {
            thisNode.firstObj = static_cast<int>(flatIndices.size());
            thisNode.objCount = static_cast<int>(nodeIndices.size());
            for (int idx : nodeIndices) {
                flatIndices.push_back(idx);
            }
            return;
        }

        // Otherwise, subdivide into 8 children
        // first find the midpoint
        const float midX = 0.5f * (thisNode.minX + thisNode.maxX);
        const float midY = 0.5f * (thisNode.minY + thisNode.maxY);
        const float midZ = 0.5f * (thisNode.minZ + thisNode.maxZ);

        // We gather each subset of nodeIndices that fully fits in each child
        std::vector<int> childSets[8];
        childSets[0].reserve(nodeIndices.size());
        childSets[1].reserve(nodeIndices.size());
        childSets[2].reserve(nodeIndices.size());
        childSets[3].reserve(nodeIndices.size());
        childSets[4].reserve(nodeIndices.size());
        childSets[5].reserve(nodeIndices.size());
        childSets[6].reserve(nodeIndices.size());
        childSets[7].reserve(nodeIndices.size());

        for (int idx : nodeIndices) {
            // sphere bounding
            const float cx = spheres[idx].center.x();
            const float cy = spheres[idx].center.y();
            const float cz = spheres[idx].center.z();
            const float r = spheres[idx].radius;

            // We'll see which child can contain it fully
            // child i bit 0 => x half, bit 1 => y half, bit 2 => z half
            // We'll check if it "fits" in a single child
            // If not, we'll keep it here (like "loose" in parent).
            int childI = -1;
            for (int c = 0; c < 8; c++) {
                // figure out child box
                const float cminX = ((c & 1) ? midX : thisNode.minX);
                const float cmaxX = ((c & 1) ? thisNode.maxX : midX);
                const float cminY = ((c & 2) ? midY : thisNode.minY);
                const float cmaxY = ((c & 2) ? thisNode.maxY : midY);
                const float cminZ = ((c & 4) ? midZ : thisNode.minZ);
                const float cmaxZ = ((c & 4) ? thisNode.maxZ : midZ);

                // check if sphere fits fully
                if ((cx - r >= cminX) && (cx + r <= cmaxX) &&
                    (cy - r >= cminY) && (cy + r <= cmaxY) &&
                    (cz - r >= cminZ) && (cz + r <= cmaxZ))
                {
                    childI = c;
                    break;
                }
            }

            if (childI >= 0) {
                childSets[childI].push_back(idx);
            }
            else {
                // doesn't fit in any child, so keep at this node
            }
        }

        // Now, the spheres that do not fit in any child remain to be stored at this node
        // find how many ended up in children
        size_t totalInChildren = 0;
        for (int c = 0; c < 8; c++) {
            totalInChildren += childSets[c].size();
        }

        std::vector<int> leftover;
        leftover.reserve(nodeIndices.size() - totalInChildren);
        // gather leftover
        // if a sphere wasn't in the child sets, it belongs to this node
        for (int idx : nodeIndices) {
            bool foundInChild = false;
            for (int c = 0; c < 8; c++) {
                for (const int x : childSets[c]) {
                    if (x == idx) {
                        foundInChild = true;
                        break;
                    }
                }
                if (foundInChild) break;
            }
            if (!foundInChild) leftover.push_back(idx);
        }

        // store leftover in this node
        thisNode.firstObj = static_cast<int>(flatIndices.size());
        thisNode.objCount = static_cast<int>(leftover.size());
        for (int idx : leftover) {
            flatIndices.push_back(idx);
        }

        // Now create child nodes if childSets[c] non-empty
        for (int c = 0; c < 8; c++) {
            if (!childSets[c].empty()) {
	            const int childNodeIdx = createNode();
                // fill child bounding box (sub box)
                FlatNode& ch = flatNodes[childNodeIdx];
                // init box
                ch.minX = ((c & 1) ? midX : thisNode.minX);
                ch.maxX = ((c & 1) ? thisNode.maxX : midX);
                ch.minY = ((c & 2) ? midY : thisNode.minY);
                ch.maxY = ((c & 2) ? thisNode.maxY : midY);
                ch.minZ = ((c & 4) ? midZ : thisNode.minZ);
                ch.maxZ = ((c & 4) ? thisNode.maxZ : midZ);
                // we don't expand further since we already know it's within
                // that sub-box.

                thisNode.children[c] = childNodeIdx;

                // recursively subdiv the child if needed
                buildNode(spheres, childSets[c], depth + 1, thisNodeIdx);
            }
            else {
                thisNode.children[c] = -1; // no child
            }
        }
    }
};

// Upload the CPU's flatNodes/flatIndices to the GPU
inline void uploadOctreeToGPU(const std::vector<FlatNode>& fnodes,
    const std::vector<int>& findices,
    GpuOctree& outGpu)
{
    outGpu.numNodes = static_cast<int>(fnodes.size());
    outGpu.numIndices = static_cast<int>(findices.size());
    if (outGpu.numNodes == 0) return; // no data

    cudaMalloc(reinterpret_cast<void**>(&outGpu.nodes), outGpu.numNodes * sizeof(FlatNode));
    cudaMalloc(reinterpret_cast<void**>(&outGpu.indices), outGpu.numIndices * sizeof(int));

    cudaMemcpy(outGpu.nodes, fnodes.data(), outGpu.numNodes * sizeof(FlatNode),
        cudaMemcpyHostToDevice);
    cudaMemcpy(outGpu.indices, findices.data(), outGpu.numIndices * sizeof(int),
        cudaMemcpyHostToDevice);
}


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

// If no hit, returns -1 for sphereIdx
struct OctreeHit {
    float t;
    int   sphereIdx;
};

__device__ bool intersectSphere(const float3& rayOrig,
    const float3& rayDir,
    float cx, float cy, float cz,
    float radius,
    float& outT)
{
    // Solve (O + tD - C)*(O + tD - C) = r^2
    // O=rayOrig, D=rayDir, C=(cx,cy,cz)
    float ox = rayOrig.x - cx;
    float oy = rayOrig.y - cy;
    float oz = rayOrig.z - cz;
    float dx = rayDir.x;
    float dy = rayDir.y;
    float dz = rayDir.z;

    float A = dx * dx + dy * dy + dz * dz;
    float B = 2.0f * (ox * dx + oy * dy + oz * dz);
    float C = ox * ox + oy * oy + oz * oz - radius * radius;

    float disc = B * B - 4.0f * A * C;
    if (disc < 0.0f) {
        return false;
    }
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

__device__ OctreeHit octreeIntersectRay(const float3& rayOrig,
    const float3& rayDir,
    const GpuOctree& oct,
    const SphereData* gpuSpheres,
    int numSpheres,
    float tMin,
    float tMax)
{
    OctreeHit result;
    result.t = -1.f;
    result.sphereIdx = -1;

    if (oct.numNodes == 0) return result;

    float3 invDir = make_float3((rayDir.x == 0.f) ? 1e32f : 1.f / rayDir.x,
        (rayDir.y == 0.f) ? 1e32f : 1.f / rayDir.y,
        (rayDir.z == 0.f) ? 1e32f : 1.f / rayDir.z);
    float closestT = tMax;

    const int maxStack = 64;
    int stackArr[maxStack];
    int stackTop = 0;
    stackArr[stackTop++] = 0; // push root node

    while (stackTop > 0) {
        int nodeId = stackArr[--stackTop];
        const FlatNode& nd = oct.nodes[nodeId];

        // Build AABB for that node
        AABB box;
        box.minX = nd.minX; box.minY = nd.minY; box.minZ = nd.minZ;
        box.maxX = nd.maxX; box.maxY = nd.maxY; box.maxZ = nd.maxZ;

        // AABB test
        if (!intersectAABB(rayOrig, invDir, box)) continue;

        // For each sphere index in this node:
        for (int i = 0; i < nd.objCount; i++) {
            int sIdx = oct.indices[nd.firstObj + i];
            if (sIdx < 0 || sIdx >= numSpheres) continue;

            // get bounding data
            float cx = gpuSpheres[sIdx].cx;
            float cy = gpuSpheres[sIdx].cy;
            float cz = gpuSpheres[sIdx].cz;
            float r = gpuSpheres[sIdx].radius;

            // do a real sphere intersection to see if this is the nearest so far
            float tCandidate;
            if (intersectSphere(rayOrig, rayDir, cx, cy, cz, r, tCandidate)) {
                if (tCandidate > tMin && tCandidate < closestT) {
                    closestT = tCandidate;
                    result.sphereIdx = sIdx;
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

    if (result.sphereIdx >= 0) {
        result.t = closestT;
    }
    return result;
}

