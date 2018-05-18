/*!
 *
 *  rand.hpp
 *  wire
 *
 *  Purpose:
 *  pointer-unique identifier
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <string>
#include <random>
#include <chrono>
#include <sstream>

#pragma once
#ifndef WIRE_RAND_HPP
#define WIRE_RAND_HPP

namespace wire
{

std::default_random_engine& get_generator (void);

void seed_generator (size_t val);

//! pointer-unique identifier
std::string puid (const void* addr);

}

#endif /* WIRE_RAND_HPP */
