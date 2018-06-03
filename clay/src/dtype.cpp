//
//  shape.cpp
//  clay
//

#include <functional>
#include <numeric>

#include "clay/dtype.hpp"
#include "clay/error.hpp"

#ifdef CLAY_DTYPE_HPP

namespace clay
{

unsigned short type_size (DTYPE type)
{
	switch (type)
	{
		case DTYPE::DOUBLE:
			return sizeof(double);
		case DTYPE::FLOAT:
			return sizeof(float);
		case DTYPE::INT8:
		case DTYPE::UINT8:
			return sizeof(int8_t);
		case DTYPE::INT16:
		case DTYPE::UINT16:
			return sizeof(int16_t);
		case DTYPE::INT32:
		case DTYPE::UINT32:
			return sizeof(int32_t);
		case DTYPE::INT64:
		case DTYPE::UINT64:
			return sizeof(int64_t);
		default:
			throw UnsupportedTypeError(type);
	}
}

template <>
DTYPE get_type<double> (void)
{
	return DTYPE::DOUBLE;
}

template <>
DTYPE get_type<float> (void)
{
	return DTYPE::FLOAT;
}

template <>
DTYPE get_type<int8_t> (void)
{
	return DTYPE::INT8;
}

template <>
DTYPE get_type<uint8_t> (void)
{
	return DTYPE::UINT8;
}

template <>
DTYPE get_type<int16_t> (void)
{
	return DTYPE::INT16;
}

template <>
DTYPE get_type<uint16_t> (void)
{
	return DTYPE::UINT16;
}

template <>
DTYPE get_type<int32_t> (void)
{
	return DTYPE::INT32;
}

template <>
DTYPE get_type<uint32_t> (void)
{
	return DTYPE::UINT32;
}

template <>
DTYPE get_type<int64_t> (void)
{
	return DTYPE::INT64;
}

template <>
DTYPE get_type<uint64_t> (void)
{
	return DTYPE::UINT64;
}

}

#endif
