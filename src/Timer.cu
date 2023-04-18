#include "Timer.cuh"

Timer::Timer() :
    last(std::chrono::high_resolution_clock::now()) {}

Timer::~Timer() {}

float Timer::duration() const {
    auto current = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = current - last;
    return d.count();
}

void Timer::update() {
    last = std::chrono::high_resolution_clock::now();
}