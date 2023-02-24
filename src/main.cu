#include <string>
#include <iostream>

#include "CudaHelper.cuh"
#include "Renderer.cuh"

bool gpu = false;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Please enter configuration file path" << std::endl;
        exit(1);
    } else if (argc > 2 && strcmp(argv[2], "--gpu") == 0)
        gpu = true;

    Renderer renderer(900, 900, argv[1]);
    renderer.render();

    return 0;
}