#include "util/error.hpp"

#ifdef UTIL_ERROR_HPP

void handle_args (std::stringstream& ss, std::string entry)
{
	ss << entry;
}

void handle_error (std::string msg)
{
	throw std::runtime_error(msg);
}

#endif
