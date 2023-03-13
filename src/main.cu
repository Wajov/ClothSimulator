#include <string>
#include <iostream>

#include "CudaHelper.cuh"
#include "Simulator.cuh"

bool gpu = false;

int main(int argc, char **argv) {
    Simulator* simulator;
    if (argc < 3) {
        std::cerr << "Please enter mode and configuration file path" << std::endl;
        exit(1);
    } else if (argc > 2 && strcmp(argv[1], "simulate") == 0) {
        if (argc > 3 && strcmp(argv[3], "--gpu") == 0)
            gpu = true;
        simulator = new Simulator(Simulate, argv[2], "");
    } else if (argc > 3 && strcmp(argv[1], "simulate_offline") == 0) {
        if (argc > 4 && strcmp(argv[4], "--gpu") == 0)
            gpu = true;
        simulator = new Simulator(SimulateOffline, argv[2], argv[3]);
    } else if (argc > 2 && strcmp(argv[1], "resume") == 0) {
        if (argc > 3 && strcmp(argv[3], "--gpu") == 0)
            gpu = true;
        simulator = new Simulator(Resume, "", argv[2]);
    } else if (argc > 2 && strcmp(argv[1], "resume_offline") == 0) {
        if (argc > 3 && strcmp(argv[3], "--gpu") == 0)
            gpu = true;
        simulator = new Simulator(ResumeOffline, "", argv[2]);
    } else if (argc > 2 && strcmp(argv[1], "replay") == 0)
        simulator = new Simulator(Replay, "", argv[2]);
    else {
        std::cerr << "Mode should be one of simulate/simulate_offline/resume/resume_offline/display" << std::endl;
        exit(1);
    }

    simulator->start();
    delete simulator;

    return 0;
}