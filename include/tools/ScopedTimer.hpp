#ifndef SCOPEDTIMER_HPP
#define SCOPEDTIMER_HPP

#include <chrono>
#include <iostream>
#include <string>

#ifdef TIMING_MODE
class ScopedTimer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    ScopedTimer(const std::string &name)
      : func_name(name), start(clock::now()) {}

    ~ScopedTimer() {
        auto stop = clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << func_name << " took " << (us / 1000.0) << " ms\n";
    }

    double elapsed_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
    }

private:
    std::string func_name;
    time_point start;
};
#else
class ScopedTimer {
public:
    ScopedTimer(const std::string&) {}
    ~ScopedTimer() {}
    double elapsed_us() const { return 0.0; }
};
#endif

#endif // SCOPEDTIMER_HPP
