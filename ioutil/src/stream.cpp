//
//  stream.cpp
//  ioutil
//

#include "ioutil/stream.hpp"

#ifdef IOUTIL_STREAM_HPP

namespace ioutil
{

Stream::Stream (void) = default;

std::string Stream::str(void) const
{
	return stream_.str();
}

Stream::operator std::string () const
{
	return stream_.str();
}

std::string Stream::operator >> (convert_to_string)
{
	return stream_.str();
}

}

#endif /* IOUTIL_STREAM_HPP */
