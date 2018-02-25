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

#include "proto/serial/tenncor.pb.h"

#pragma once
#ifndef TENNCOR_TENS_TYPE_HPP
#define TENNCOR_TENS_TYPE_HPP

namespace nnet
{

class varptr;

#define TENS_TYPE tenncor::tensor_proto::tensor_t

#define BAD_T tenncor::tensor_proto::BAD
#define DOUBLE tenncor::tensor_proto::DOUBLE
#define FLOAT tenncor::tensor_proto::FLOAT
#define INT8 tenncor::tensor_proto::INT8
#define UINT8 tenncor::tensor_proto::UINT8
#define INT16 tenncor::tensor_proto::INT16
#define UINT16 tenncor::tensor_proto::UINT16
#define INT32 tenncor::tensor_proto::INT32
#define UINT32 tenncor::tensor_proto::UINT32
#define INT64 tenncor::tensor_proto::INT64
#define UINT64 tenncor::tensor_proto::UINT64

unsigned short type_size (TENS_TYPE type);

template <typename T>
TENS_TYPE get_type (void)
{
	throw std::exception(); // todo: make type error
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
