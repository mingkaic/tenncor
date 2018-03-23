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

#include "proto/serial/data.pb.h"

#pragma once
#ifndef TENNCOR_TENS_TYPE_HPP
#define TENNCOR_TENS_TYPE_HPP

namespace nnet
{

#define TENS_TYPE tenncor::tensor_t
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

}

#endif /* TENNCOR_TENS_TYPE_HPP */
