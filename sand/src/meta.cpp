#include <cstring>

#include "sand/meta.hpp"

#ifdef SAND_META_HPP

std::string Meta::to_string (void) const
{
	return shape_.to_string() + ":" + name_type(type_);
}

size_t Meta::nbytes (void) const
{
	return shape_.n_elems() * type_size(type_);
}

MetaEncoder::MetaEncoder (SCODE code) : code_(code)
{
	std::memset(data_, 0, NHash);
}

MetaEncoder::MetaEncoder (const MetaEncoder& other) : code_(other.code_)
{
	std::memcpy(data_, other.data_, NHash);
}

MetaEncoder::MetaEncoder (MetaEncoder&& other) = default;

MetaEncoder& MetaEncoder::operator = (const MetaEncoder& other)
{
	if (this != &other)
	{
		code_ = other.code_;
		std::memcpy(data_, other.data_, NHash);
	}
	return *this;
}

MetaEncoder& MetaEncoder::operator = (MetaEncoder&& other) = default;

MetaEncoder::operator std::string(void)
{
	std::string out(1 + NHash, 0);
	out[0] = code_;
	std::memcpy(&out[1], data_, NHash);
	return out;
}

#endif
