#ifndef SEPARATION_HELPER_CUH
#define SEPARATION_HELPER_CUH

#include <vector>

#include "MathHelper.cuh"
#include "Node.cuh"
#include "Face.cuh"
#include "Intersection.cuh"
#include "Cloth.cuh"

const int MAX_SEPARATION_ITERATION = 100;

int majorAxis(const Vector3f& v);
bool facePlaneIntersection(const Face* face, const Face* plane, Vector3f& b0, Vector3f& b1);
bool intersectionMidpoint(const Face* face0, const Face* face1, Vector3f& b0, Vector3f& b1);
Vector3f oldPosition(const Face* face, const Vector3f& b, const std::vector<Cloth*>& cloths, const std::vector<Mesh*>& oldMeshes);
void clearVertexFaceDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
void clearEdgeEdgeDistance(const Face* face0, const Face* face1, const Vector3f& d, float& maxDist, Vector3f& b0, Vector3f& b1);
void farthestPoint(const Face* face0, const Face* face1, const Vector3f& d, Vector3f& b0, Vector3f& b1);

#endif