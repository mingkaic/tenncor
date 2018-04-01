/*!
 *
 *  type.hpp
 *  cnnet
 *
 *  Purpose:
 *  type information for tenncor
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/utils/error.hpp"

#pragma once
#ifndef TENNCOR_TENS_TYPE_HPP
#define TENNCOR_TENS_TYPE_HPP

namespace nnet
{

using TYPE2VAL = std::function<std::string(TENS_TYPE)>;

#define _TYPE_SENTINEL tenncor::_TYPE_SENTINEL

#define BAD_T tenncor::BAD
#define DOUBLE tenncor::DOUBLE
#define FLOAT tenncor::FLOAT
#define INT8 tenncor::INT8
#define UINT8 tenncor::UINT8
#define INT16 tenncor::INT16
#define UINT16 tenncor::UINT16
#define INT32 tenncor::INT32
#define UINT32 tenncor::UINT32
#define INT64 tenncor::INT64
#define UINT64 tenncor::UINT64

unsigned short type_size (TENS_TYPE type);

template <typename T>
TENS_TYPE get_type (void)
{
	return BAD_T;
}

template <>
TENS_TYPE get_type<double> (void);

template <>
TENS_TYPE get_type<float> (void);

template <>
TENS_TYPE get_type<int8_t> (void);

template <>
TENS_TYPE get_type<uint8_t> (void);

template <>
TENS_TYPE get_type<int16_t> (void);

template <>
TENS_TYPE get_type<uint16_t> (void);

template <>
TENS_TYPE get_type<int32_t> (void);

template <>
TENS_TYPE get_type<uint32_t> (void);

template <>
TENS_TYPE get_type<int64_t> (void);

template <>
TENS_TYPE get_type<uint64_t> (void);

#define CONVERT(out, in, otype, itype) \
itype* inptr = (itype*) in; \
out = std::vector<otype>(inptr, inptr + n);

template <typename OT>
std::vector<OT> type_convert(void* ptr, size_t n, TENS_TYPE itype)
{
	TENS_TYPE otype = get_type<OT>();
	assert(otype != BAD_T);
	if (otype == itype)
	{
		OT* optr = (OT*) ptr;
		return std::vector<OT>(optr, optr + n);
	}
	std::vector<OT> out;
	switch (itype)
	{
		case DOUBLE:
		{
			CONVERT(out, ptr, OT, double)
		}
		break;
		case FLOAT:
		{
			CONVERT(out, ptr, OT, float)
		}
		break;
		case INT8:
		{
			CONVERT(out, ptr, OT, int8_t)
		}
		break;
		case UINT8:
		{
			CONVERT(out, ptr, OT, uint8_t)
		}
		break;
		case INT16:
		{
			CONVERT(out, ptr, OT, int16_t)
		}
		break;
		case UINT16:
		{
			CONVERT(out, ptr, OT, uint16_t)
		}
		break;
		case INT32:
		{
			CONVERT(out, ptr, OT, int32_t)
		}
		break;
		case UINT32:
		{
			CONVERT(out, ptr, OT, uint32_t)
		}
		break;
		case INT64:
		{
			CONVERT(out, ptr, OT, int64_t)
		}
		break;
		case UINT64:
		{
			CONVERT(out, ptr, OT, uint64_t)
		}
		break;
		default:
			throw nnutils::unsupported_type_error(itype);
	}
	return out;
}

namespace type
{

std::string maxval (TENS_TYPE type);

std::string minval (TENS_TYPE type);

std::string zeroval (TENS_TYPE type);

}

}

#endif /* TENNCOR_TENS_TYPE_HPP */
