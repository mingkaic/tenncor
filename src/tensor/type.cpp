/*!
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/type.hpp"

#ifdef TENNCOR_TENS_TYPE_HPP

namespace nnet
{

unsigned short type_size (TENS_TYPE type)
{
	switch (type)
	{
		case DOUBLE:
			return sizeof(double);
		case FLOAT:
			return sizeof(float);
		// asserts that signed and unsigned 
		// always has the same bit width
		case INT8:
		case UINT8:
			return sizeof(int8_t);
		case INT16:
		case UINT16:
			return sizeof(int16_t);
		case INT32:
		case UINT32:
			return sizeof(int32_t);
		case INT64:
		case UINT64:
			return sizeof(int64_t);
		default:
			throw std::exception(); // todo: add type error
	}
}

template <>
TENS_TYPE get_type<double> (void)
{
	return DOUBLE;
}

template <>
TENS_TYPE get_type<float> (void)
{
	return FLOAT;
}

template <>
TENS_TYPE get_type<int8_t> (void)
{
	return INT8;
}

template <>
TENS_TYPE get_type<uint8_t> (void)
{
	return UINT8;
}

template <>
TENS_TYPE get_type<int16_t> (void)
{
	return INT16;
}

template <>
TENS_TYPE get_type<uint16_t> (void)
{
	return UINT16;
}

template <>
TENS_TYPE get_type<int32_t> (void)
{
	return INT32;
}

template <>
TENS_TYPE get_type<uint32_t> (void)
{
	return UINT32;
}

template <>
TENS_TYPE get_type<int64_t> (void)
{
	return INT64;
}

template <>
TENS_TYPE get_type<uint64_t> (void)
{
	return UINT64;
}

}

#endif
