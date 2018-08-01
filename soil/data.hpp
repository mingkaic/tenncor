#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

#include "soil/shape.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"

#ifndef DATA_HPP
#define DATA_HPP

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

std::string name_type (DTYPE type);

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

uint8_t type_size (DTYPE type);

struct DataSource
{
	DataSource (DTYPE type, NElemT n);

	DataSource (char* data, DTYPE type, NElemT n);

	template <typename T>
	operator T* (void) const
	{
		DTYPE dtype = get_type<T>();
		if (dtype != type_)
		{
			handle_error("incompatible type",
				ErrArg<std::string>("in_type", name_type(dtype)),
				ErrArg<DTYPE>("out_type", type_));
		}
		return (T*) data_.get();
	}

	char* data (void) const;

	DTYPE type (void) const;

protected:
	std::shared_ptr<char> data_;
	DTYPE type_;
};

#endif /* DATA_HPP */
