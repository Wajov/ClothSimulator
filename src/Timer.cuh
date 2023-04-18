#ifndef TIMER_CUH
#define TIMER_CUH

#include <chrono>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> last;

public:
    Timer();
    ~Timer();
    float duration() const;
    void update();
};

#endif