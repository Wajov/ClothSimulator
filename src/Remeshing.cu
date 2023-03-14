#include "Remeshing.cuh"

Remeshing::Remeshing(const Json::Value& json) {
    refineAngle = parseFloat(json["refine_angle"], INFINITY);
    refineVelocity = parseFloat(json["refine_velocity"], INFINITY);
    refineCompression = parseFloat(json["refine_compression"], INFINITY);
    ribStiffening = parseFloat(json["rib_stiffening"], 1.0f);

    if (json["size"].isNull()) {
        sizeMin = -INFINITY;
        sizeMax = INFINITY;
    } else {
        sizeMin = parseFloat(json["size"][0]);
        sizeMax = parseFloat(json["size"][1]);
    }
    aspectMin = parseFloat(json["aspect_min"], -INFINITY);
    flipThreshold = parseFloat(json["flip_threshold"], 1e-2f); 
}

Remeshing::~Remeshing() {}