#include <string>
#include <iostream>

#include "Renderer.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Please enter configuration file path" << std::endl;
        exit(1);
    }

    Renderer renderer(900, 900, argv[1]);
    renderer.render();

    return 0;
}