#include <chrono>
#include <sstream>
#include <ios>

#include "util/rand.hpp"

#ifdef UTIL_RAND_HPP

namespace util
{

ENGINE& get_engine (void)
{
	static ENGINE engine;
	return engine;
}

std::string make_uid (void* ptr, ENGINE& engine)
{
	static std::uniform_int_distribution<size_t> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) ptr;

	for (size_t i = 0; i < 16; i++)
	{
		size_t token = tok_dist(engine);
		ss << std::hex << token;
	}
	return ss.str();
}

}

#endif
