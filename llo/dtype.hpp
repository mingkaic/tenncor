///
/// dtype.hpp
/// llo
///
/// Purpose:
/// Enumerate tensor data types
///

#include <string>

#ifndef LLO_DTYPE_HPP
#define LLO_DTYPE_HPP

namespace llo
{

/// Enumerated representation of data types
enum DTYPE
{
	BAD = 0, /// Invalid data type
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

/// Return the string name of input DTYPE
std::string nametype (DTYPE type);

/// Return the byte size of input DTYPE
uint8_t type_size (DTYPE type);

/// Return the DTYPE of type in template
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

#endif // LLO_DTYPE_HPP
