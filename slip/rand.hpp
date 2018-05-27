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
#ifndef SLIP_RAND_HPP
#define SLIP_RAND_HPP

namespace slip
{

std::default_random_engine& get_generator (void);

void seed_generator (size_t val);

//! pointer-unique identifier
std::string puid (const void* addr);

}

#endif /* SLIP_RAND_HPP */
