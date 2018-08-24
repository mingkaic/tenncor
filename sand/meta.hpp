#include "sand/shape.hpp"
#include "sand/type.hpp"

#ifndef SAND_META_HPP
#define SAND_META_HPP

struct Meta
{
	std::string to_string (void) const;

	size_t nbytes (void) const;

	Shape shape_;
	DTYPE type_;
};

enum SCODE
{
	ELEM = 0,
	TSHAPE,
	MATSHAPE,
	TCAST,
	GROUP,
	NELEMSPRE,
	NDIMSPRE,
	BINOPRE,
	REDUCEPRE,
};

struct MetaEncoder
{
	static const uint8_t NHash = 4;

	using MetaData = uint8_t[NHash];

	MetaEncoder (SCODE code);

	MetaEncoder (const MetaEncoder& other);

	MetaEncoder (MetaEncoder&& other);

	MetaEncoder& operator = (const MetaEncoder& other);

	MetaEncoder& operator = (MetaEncoder&& other);

	operator std::string(void);

	SCODE code_;
	MetaData data_;
};

#endif /* SAND_META_HPP */
