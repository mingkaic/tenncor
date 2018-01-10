#include <iostream>

#include "include/utils/utils.hpp"

#ifdef TENNCOR_UTILS_HPP

namespace nnutils
{

formatter::formatter (void) {}

formatter::~formatter (void) {}

std::string formatter::str (void) const
{
	return stream_.str();
}

formatter::operator std::string () const
{
	return stream_.str();
}

std::string formatter::operator >> (convert_to_string)
{
	return stream_.str();
}

boost::uuids::uuid uuid (void)
{
	static boost::uuids::random_generator uuid_gen(get_boost_generator());
	return uuid_gen();
}

std::default_random_engine& get_generator (void)
{
	static std::default_random_engine common_gen(std::time(nullptr));
	return common_gen;
}

boost::mt19937& get_boost_generator (void)
{
	static boost::mt19937 boost_gen(std::time(nullptr));
	return boost_gen;
}

void seed_generator (size_t val)
{
	get_generator().seed(val);
	get_boost_generator().seed(val);
}

}

#endif