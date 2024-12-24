#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

// Axis-aligned bounding box (AABB)
struct AABB {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;

    AABB()
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

    // Simple slab test for ray-box intersection
    bool intersectAABB(const float* rayOrig, const float* rayDirInv) const {
        // rayOrig = [ox, oy, oz]
        // rayDirInv = [1/dx, 1/dy, 1/dz], assuming dx,dy,dz != 0
        float t1 = (minX - rayOrig[0]) * rayDirInv[0];
        float t2 = (maxX - rayOrig[0]) * rayDirInv[0];
        float tmin = std::min(t1, t2);
        float tmax = std::max(t1, t2);

        t1 = (minY - rayOrig[1]) * rayDirInv[1];
        t2 = (maxY - rayOrig[1]) * rayDirInv[1];
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));

        t1 = (minZ - rayOrig[2]) * rayDirInv[2];
        t2 = (maxZ - rayOrig[2]) * rayDirInv[2];
        tmin = std::max(tmin, std::min(t1, t2));
        tmax = std::min(tmax, std::max(t1, t2));

        // If tmax < tmin => no intersection
        return (tmax > std::max(tmin, 0.0f));
    }
};

// Octree node that stores spheres
class OctreeNode {
public:
    AABB bounds;
    std::vector<sphere> objects;
    std::unique_ptr<OctreeNode> children[8]; // 8 octants

    // Insert a sphere into this node
    void insert(const sphere& s, int maxObjectsPerNode, int maxDepth, int currentDepth = 0) {
        // Expand this node's bounding box
        bounds.expand(s);

        // If this node can hold sphere, or if we reached max depth, store it here
        if (objects.size() < (size_t)maxObjectsPerNode || currentDepth >= maxDepth) {
            objects.push_back(s);
            return;
        }

        // Otherwise subdivide if we haven't yet
        if (!children[0]) {
            subdivide();
        }

        // Attempt to place sphere into exactly one child if it fits fully
        for (int i = 0; i < 8; i++) {
            if (childContainsSphere(children[i]->bounds, s)) {
                children[i]->insert(s, maxObjectsPerNode, maxDepth, currentDepth + 1);
                return;
            }
        }

        // If sphere doesn't fully fit into any single child, keep it here
        objects.push_back(s);
    }
    
    bool traverseRay(const float* rayOrig, const float* rayDir,
        const float* rayDirInv, float& outHitT) const
    {
        //  - returns true if ray hits at least one sphere in this subtree
	    //    and updates outHitT with closest intersection distance
	    //  - returns false if there's no hit.

        // Quickly exit if ray doesn't intersect this node's AABB
        if (!bounds.intersectAABB(rayOrig, rayDirInv)) {
            return false;
        }

        bool hitAny = false;
        float closestT = outHitT;

        // Check for intersection against local spheres
        for (const auto& s : objects) {
            float tHit;
            if (intersectSphere(rayOrig, rayDir, s, tHit)) {
                if (tHit < closestT) {
                    closestT = tHit;
                    hitAny = true;
                }
            }
        }

        // Recurse into children
        for (int i = 0; i < 8; i++) {
            if (children[i]) {
                float tChild = closestT; // pass in current best
                if (children[i]->traverseRay(rayOrig, rayDir, rayDirInv, tChild)) {
                    if (tChild < closestT) {
                        closestT = tChild;
                        hitAny = true;
                    }
                }
            }
        }

        if (hitAny) {
            outHitT = closestT;
        }
        return hitAny;
    }

private:
    // Subdivide node into 8 children
    void subdivide() {
        float midX = 0.5f * (bounds.minX + bounds.maxX);
        float midY = 0.5f * (bounds.minY + bounds.maxY);
        float midZ = 0.5f * (bounds.minZ + bounds.maxZ);

        for (int i = 0; i < 8; i++) {
            AABB childBox;
            childBox.minX = (i & 1) ? midX : bounds.minX;
            childBox.maxX = (i & 1) ? bounds.maxX : midX;
            childBox.minY = (i & 2) ? midY : bounds.minY;
            childBox.maxY = (i & 2) ? bounds.maxY : midY;
            childBox.minZ = (i & 4) ? midZ : bounds.minZ;
            childBox.maxZ = (i & 4) ? bounds.maxZ : midZ;

            children[i] = std::make_unique<OctreeNode>();
            children[i]->bounds = childBox;
        }
    }

    // Check whether 'childBox' fully contains sphere 's'
    bool childContainsSphere(const AABB& childBox, const sphere& s) const {
        float cx = s.center.x();
        float cy = s.center.y();
        float cz = s.center.z();
        float r = s.radius;
        return (cx - r >= childBox.minX && cx + r <= childBox.maxX &&
            cy - r >= childBox.minY && cy + r <= childBox.maxY &&
            cz - r >= childBox.minZ && cz + r <= childBox.maxZ);
    }

    // Basic sphere-ray intersection, returning t-value in outT
    bool intersectSphere(const float* rayOrig, const float* rayDir,
        const sphere& s, float& outT) const
    {
        // Solve (O + tD - C)*(O + tD - C) = r^2
        // O=rayOrig, D=rayDir, C=s.center
        float ox = rayOrig[0] - s.center.x();
        float oy = rayOrig[1] - s.center.y();
        float oz = rayOrig[2] - s.center.z();
        float dx = rayDir[0];
        float dy = rayDir[1];
        float dz = rayDir[2];
        float r = s.radius;

        float A = dx * dx + dy * dy + dz * dz;
        float B = 2.0f * (ox * dx + oy * dy + oz * dz);
        float C = ox * ox + oy * oy + oz * oz - r * r;

        float disc = B * B - 4.0f * A * C;
        if (disc < 0.0f) {
            return false;
        }
        float sqrtDisc = std::sqrt(disc);
        float t1 = (-B - sqrtDisc) / (2.0f * A);
        float t2 = (-B + sqrtDisc) / (2.0f * A);

        // We want smallest t >= 0
        float tHit = -1.0f;
        if (t1 > 0.0f && t2 > 0.0f) tHit = std::min(t1, t2);
        else if (t1 > 0.0f)        tHit = t1;
        else if (t2 > 0.0f)        tHit = t2;

        if (tHit < 0.0f) {
            return false; // behind origin
        }
        outT = tHit;
        return true;
    }
};

// Simple wrapper for octree
class Octree {
public:
    Octree(int maxObjPerNode = 4, int maxD = 8)
        : maxObjectsPerNode(maxObjPerNode), maxDepth(maxD)
    {
        root = std::make_unique<OctreeNode>();
    }

    // Build octree from a list of spheres
    void build(const std::vector<sphere>& spheres) {
        for (auto& s : spheres) {
            root->insert(s, maxObjectsPerNode, maxDepth, 0);
        }
    }

    // Minimal traversal: return true if ray hits anything.
    // outT will be updated with nearest intersection distance on hit.
    bool intersect(const float* rayOrig, const float* rayDir, float& outT) const {
        // Precompute 1/d for slab test
        float rayDirInv[3];
        for (int i = 0; i < 3; i++) {
            rayDirInv[i] = (rayDir[i] == 0.0f) ? 1e32f : 1.0f / rayDir[i];
        }
        outT = std::numeric_limits<float>::max();
        return root->traverseRay(rayOrig, rayDir, rayDirInv, outT);
    }

private:
    std::unique_ptr<OctreeNode> root;
    int maxObjectsPerNode;
    int maxDepth;
};