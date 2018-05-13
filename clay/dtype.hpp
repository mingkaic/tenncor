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

// attempt to convert in vector of template type to otype in out string, return false if fail
template <typename T>
bool convert (std::string& out, DTYPE otype, const std::vector<T>& in)
{
	if (DTYPE::BAD == otype)
	{
		return false;
	}
	unsigned short bsize = type_size(otype);
	switch (otype)
	{
		case DTYPE::DOUBLE:
		{
			std::vector<double> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::FLOAT:
		{
			std::vector<float> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::INT8:
		{
			std::vector<int8_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::UINT8:
		{
			std::vector<uint8_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::INT16:
		{
			std::vector<int16_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::UINT16:
		{
			std::vector<uint16_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::INT32:
		{
			std::vector<int32_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::UINT32:
		{
			std::vector<uint32_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::INT64:
		{
			std::vector<int64_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		case DTYPE::UINT64:
		{
			std::vector<uint64_t> temp(in.begin(), in.end());
			out = std::string((char*) &temp[0], temp.size() * bsize);
		}
		break;
		default:
			return false; // unsupported type
	}
	return true;
}

// attempt to convert in string of itype to out vector, return false if fail
template <typename T>
bool convert (std::vector<T>& out, const std::string& in, DTYPE itype)
{
	if (DTYPE::BAD == itype)
	{
		return false;
	}
	size_t n = in.size() / type_size(itype);
	switch (itype)
	{
		case DTYPE::DOUBLE:
		{
			const double* ptr = (const double*) in.c_str();
			std::vector<double> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::FLOAT:
		{
			const float* ptr = (const float*) in.c_str();
			std::vector<float> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::INT8:
		{
			const int8_t* ptr = (const int8_t*) in.c_str();
			std::vector<int8_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::UINT8:
		{
			const uint8_t* ptr = (const uint8_t*) in.c_str();
			std::vector<uint8_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::INT16:
		{
			const int16_t* ptr = (const int16_t*) in.c_str();
			std::vector<int16_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::UINT16:
		{
			const uint16_t* ptr = (const uint16_t*) in.c_str();
			std::vector<uint16_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::INT32:
		{
			const int32_t* ptr = (const int32_t*) in.c_str();
			std::vector<int32_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::UINT32:
		{
			const uint32_t* ptr = (const uint32_t*) in.c_str();
			std::vector<uint32_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::INT64:
		{
			const int64_t* ptr = (const int64_t*) in.c_str();
			std::vector<int64_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		case DTYPE::UINT64:
		{
			const uint64_t* ptr = (const uint64_t*) in.c_str();
			std::vector<uint64_t> temp(ptr, ptr + n);
			out = std::vector<T>(temp.begin(), temp.end());
		}
		break;
		default:
			return false; // unsupported type
	}
	return true;
}

// attempt to convert in string of itype to otype in out string, return false if fail
bool convert (std::string& out, DTYPE otype,
	const std::string& in, DTYPE itype);

}

#endif /* CLAY_DTYPE_HPP */
