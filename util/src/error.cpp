#include "util/error.hpp"

#ifdef ERROR_HPP

iErrArg::operator std::string (void) const
{
	return to_string();
}

void handle_args (std::stringstream& ss, std::string entry)
{
	ss << entry;
}

void handle_error (std::string msg)
{
	throw std::runtime_error(msg);
}

#endif
