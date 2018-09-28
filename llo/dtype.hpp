/*!
 *
 *  dtype.hpp
 *  llo
 *
 *  Purpose:
 *  define enumerations for representing types
 *
 */

#include <string>

#ifndef LLO_DTYPE_HPP
#define LLO_DTYPE_HPP

namespace llo
{

/*! Enumerated representation of data types */
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

/*! Convert the DTYPE to string */
std::string name_type (DTYPE type);

/*! Get size of enumerate type in bytes */
uint8_t type_size (DTYPE type);

/*! Get enumerate type specified in template */
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

#endif /* LLO_DTYPE_HPP */
