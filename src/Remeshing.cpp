#include "Remeshing.hpp"

Remeshing::Remeshing(const Json::Value& json) {
    refineAngle = parseFloat(json["refine_angle"], INFINITY);
    refineCompression = parseFloat(json["refine_compression"], INFINITY);
    refineVelocity = parseFloat(json["refine_velocity"], INFINITY);
    if (json["size"].isNull()) {
        sizeMin = -INFINITY;
        sizeMax = INFINITY;
    } else {
        sizeMin = json["size"][0].asFloat();
        sizeMax = json["size"][1].asFloat();
    }
    aspectMin = parseFloat(json["aspect_min"], -INFINITY);
}

Remeshing::~Remeshing() {}