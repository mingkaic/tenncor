//
// Created by Mingkai Chen on 2018-01-12.
//

#include <iostream>
#include "wire/rand.hpp"

#ifdef WIRE_RAND_HPP

namespace wire
{

static size_t get_seed (void)
{
	size_t seed = std::time(nullptr);
#ifdef CALLOUT_SEED
	std::cout << "initiating generator with seed " << seed << std::endl;
#endif /* CALLOUT_SEED */

	return seed;
}

std::default_random_engine& get_generator (void)
{
	static std::default_random_engine common_gen(get_seed());
	return common_gen;
}

void seed_generator (size_t val)
{
	get_generator().seed(val);
}

//! pointer-unique identifier
std::string puid (const void* addr)
{
	static std::uniform_int_distribution<size_t> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) addr;

	for (size_t i = 0; i < 16; i++)
	{
		size_t token = tok_dist(get_generator());
		ss << std::hex << token;
	}
	return ss.str();
}

}

#endif
