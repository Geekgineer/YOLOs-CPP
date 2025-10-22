// ScopedTimer.hpp
#ifndef SCOPEDTIMER_HPP
#define SCOPEDTIMER_HPP

/**
 * @file ScopedTimer.hpp
 * @brief Header file for timing utilities.
 * 
 * This file defines the ScopedTimer class, which measures the duration of a 
 * code block for performance profiling. When TIMING_MODE is defined, it records
 * the time taken for execution and prints it to the standard output upon 
 * destruction; otherwise, it provides an empty implementation.
 */

#include <chrono>
#include <iostream>
#include <string>
#include "tools/Config.hpp" // Include the config file to access the flags

#ifdef TIMING_MODE
class ScopedTimer {
public:
    /**
     * @brief Constructs a ScopedTimer to measure the duration of a named code block.
     * @param name The name of the code block being timed.
     */
    ScopedTimer(const std::string &name) 
        : func_name(name), start(std::chrono::high_resolution_clock::now()) {}
    
    /**
     * @brief Destructor that calculates and prints the elapsed time.
     */
    ~ScopedTimer() {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        std::cout << func_name << " took " << duration.count() << " milliseconds." << std::endl;
    }

private:
    std::string func_name; ///< The name of the timed function.
    std::chrono::time_point<std::chrono::high_resolution_clock> start; ///< Start time point.
};
#else
class ScopedTimer {
public:
    ScopedTimer(const std::string &name) {}
    ~ScopedTimer() {}
};
#endif // TIMING_MODE

#endif // SCOPEDTIMER_HPP