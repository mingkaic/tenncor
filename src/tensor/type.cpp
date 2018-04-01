/*!
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/type.hpp"
#include "include/utils/error.hpp"

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
		// always has the same size
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
			throw nnutils::unsupported_type_error(type);
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

std::string type_convert (void* ptr, size_t n, TENS_TYPE otype, TENS_TYPE itype)
{
	assert(otype != BAD_T && itype != BAD_T);
	size_t outbytes = n * type_size(otype);
	if (otype == itype)
	{
		return std::string((char*) ptr, outbytes);
	}
	switch (otype)
	{
		case DOUBLE:
			return std::string((char*) &type_convert<double>(ptr, n, itype)[0], outbytes);
		case FLOAT:
			return std::string((char*) &type_convert<float>(ptr, n, itype)[0], outbytes);
		case INT8:
			return std::string((char*) &type_convert<int8_t>(ptr, n, itype)[0], outbytes);
		case UINT8:
			return std::string((char*) &type_convert<uint8_t>(ptr, n, itype)[0], outbytes);
		case INT16:
			return std::string((char*) &type_convert<int16_t>(ptr, n, itype)[0], outbytes);
		case UINT16:
			return std::string((char*) &type_convert<uint16_t>(ptr, n, itype)[0], outbytes);
		case INT32:
			return std::string((char*) &type_convert<int32_t>(ptr, n, itype)[0], outbytes);
		case UINT32:
			return std::string((char*) &type_convert<uint32_t>(ptr, n, itype)[0], outbytes);
		case INT64:
			return std::string((char*) &type_convert<int64_t>(ptr, n, itype)[0], outbytes);
		case UINT64:
			return std::string((char*) &type_convert<uint64_t>(ptr, n, itype)[0], outbytes);
		default:
			throw nnutils::unsupported_type_error(itype);
	}
}

}

#endif
