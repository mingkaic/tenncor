#include <chrono>
#include <sstream>
#include <ios>

#include "util/rand.hpp"

#ifdef RAND_HPP

ENGINE& get_engine (void)
{
	static ENGINE engine;
	return engine;
}

std::string make_uid (void* ptr)
{
	static std::uniform_int_distribution<size_t> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) ptr;

	for (size_t i = 0; i < 16; i++)
	{
		size_t token = tok_dist(get_engine());
		ss << std::hex << token;
	}
	return ss.str();
}

#endif
