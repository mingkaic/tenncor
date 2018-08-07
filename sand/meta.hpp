#include "sand/shape.hpp"
#include "sand/type.hpp"

#ifndef SAND_META_HPP
#define SAND_META_HPP

struct Meta
{
	bool compatible (Meta& other) const
	{
		uint8_t rank = std::min(shape_.n_rank(), other.shape_.n_rank());
		return other.shape_.compatible_before(shape_, rank) &&
			other.type_ == type_;
	}

	std::string to_string (void) const
	{
		return shape_.to_string() + ":" + name_type(type_);
	}

	size_t nbytes (void) const
	{
		return shape_.n_elems() * type_size(type_);
	}

	Shape shape_;
	DTYPE type_;
};

enum SCODE
{
	ELEM = 0,
	TSHAPE,
	MATSHAPE,
};

struct MetaEncoder
{
	static const int NHash = 4;

	using MetaData = uint8_t[NHash];

	MetaEncoder (SCODE code) : code_(code)
	{
		std::memset(data_, 0, NHash);
	}

	MetaEncoder (const MetaEncoder& other) : code_(other.code_)
	{
		std::memcpy(data_, other.data_, NHash);
	}

	MetaEncoder (MetaEncoder&& other) = default;

	MetaEncoder& operator = (const MetaEncoder& other)
	{
		if (this != &other)
		{
			code_ = other.code_;
			std::memcpy(data_, other.data_, NHash);
		}
		return *this;
	}

	MetaEncoder& operator = (MetaEncoder&& other) = default;

	operator std::string(void)
	{
		std::string out(1 + NHash, 0);
		out[0] = code_;
		std::memcpy(&out[1], data_, NHash);
		return out;
	}

	SCODE code_;
	MetaData data_;
};

#endif /* SAND_META_HPP */
