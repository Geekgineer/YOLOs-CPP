// Debug.hpp
#ifndef DEBUG_HPP
#define DEBUG_HPP

/**
 * @file Debug.hpp
 * @brief Header file for debugging utilities.
 * 
 * This file provides macros to enable or disable debug printing based on the
 * configuration. When DEBUG_MODE is defined, debug messages are printed to the
 * standard output; otherwise, they are ignored.
 */

// Include necessary libraries
#include <iostream>
#include "tools/Config.hpp" // Include the config file to access the flags

#ifdef DEBUG_MODE
#define DEBUG_PRINT(x) std::cout << x << std::endl;
#else
#define DEBUG_PRINT(x)
#endif

#endif // DEBUG_HPP
