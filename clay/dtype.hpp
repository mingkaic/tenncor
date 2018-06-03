/*!
 *
 *  dtype.hpp
 *  clay
 *
 *  Purpose:
 *  dtype defines DTYPE enum
 *
 *  Created by Mingkai Chen on 2018-05-09.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <exception>

#pragma once
#ifndef CLAY_DTYPE_HPP
#define CLAY_DTYPE_HPP

namespace clay
{

enum DTYPE
{
	BAD = 0,
	DOUBLE,
	FLOAT,
	INT8,
	INT16,
	INT32,
	INT64,
	UINT8,
	UINT16,
	UINT32,
	UINT64,
	_SENTINEL
};

unsigned short type_size (DTYPE type);

template <typename T>
DTYPE get_type (void)
{
	return DTYPE::BAD;
}

template <>
DTYPE get_type<double> (void);

template <>
DTYPE get_type<float> (void);

template <>
DTYPE get_type<int8_t> (void);

template <>
DTYPE get_type<uint8_t> (void);

template <>
DTYPE get_type<int16_t> (void);

template <>
DTYPE get_type<uint16_t> (void);

template <>
DTYPE get_type<int32_t> (void);

template <>
DTYPE get_type<uint32_t> (void);

template <>
DTYPE get_type<int64_t> (void);

template <>
DTYPE get_type<uint64_t> (void);

}

#endif /* CLAY_DTYPE_HPP */
