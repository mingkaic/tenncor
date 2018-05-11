//
//  shape.cpp
//  clay
//

#include <functional>
#include <numeric>

#include "clay/dtype.hpp"

#ifdef TENSOR_DTYPE_HPP

namespace clay
{

unsigned short type_size (DTYPE type)
{
	switch (type)
	{
		case DTYPE::DOUBLE:
			return sizeof(double);
		case DTYPE::FLOAT:
			return sizeof(float);
		case DTYPE::INT8:
		case DTYPE::UINT8:
			return sizeof(int8_t);
		case DTYPE::INT16:
		case DTYPE::UINT16:
			return sizeof(int16_t);
		case DTYPE::INT32:
		case DTYPE::UINT32:
			return sizeof(int32_t);
		case DTYPE::INT64:
		case DTYPE::UINT64:
			return sizeof(int64_t);
		default:
			throw std::exception(); // todo: add context
	}
}

template <>
DTYPE get_type<double> (void)
{
	return DTYPE::DOUBLE;
}

template <>
DTYPE get_type<float> (void)
{
	return DTYPE::FLOAT;
}

template <>
DTYPE get_type<int8_t> (void)
{
	return DTYPE::INT8;
}

template <>
DTYPE get_type<uint8_t> (void)
{
	return DTYPE::UINT8;
}

template <>
DTYPE get_type<int16_t> (void)
{
	return DTYPE::INT16;
}

template <>
DTYPE get_type<uint16_t> (void)
{
	return DTYPE::UINT16;
}

template <>
DTYPE get_type<int32_t> (void)
{
	return DTYPE::INT32;
}

template <>
DTYPE get_type<uint32_t> (void)
{
	return DTYPE::UINT32;
}

template <>
DTYPE get_type<int64_t> (void)
{
	return DTYPE::INT64;
}

template <>
DTYPE get_type<uint64_t> (void)
{
	return DTYPE::UINT64;
}

// attempt to convert in string of itype to otype in out string, return false if fail
bool convert (std::string& out, DTYPE otype, const std::string& in, DTYPE itype)
{
	if (otype == DTYPE::BAD || itype == DTYPE::BAD)
	{
		return false;
	}
	if (otype == itype)
	{
		out = in;
		return true;
	}
	size_t n = in.size() * type_size(itype);
	bool success;
	switch (itype)
	{
		case DTYPE::DOUBLE:
		{
			double* iptr = (double*) &in[0];
			success = convert(out, otype, std::vector<double>(iptr, iptr + n));
		}
		break;
		case DTYPE::FLOAT:
		{
			float* iptr = (float*) &in[0];
			success = convert(out, otype, std::vector<float>(iptr, iptr + n));
		}
		break;
		case DTYPE::INT8:
		{
			int8_t* iptr = (int8_t*) &in[0];
			success = convert(out, otype, std::vector<int8_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::UINT8:
		{
			uint8_t* iptr = (uint8_t*) &in[0];
			success = convert(out, otype, std::vector<uint8_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::INT16:
		{
			int16_t* iptr = (int16_t*) &in[0];
			success = convert(out, otype, std::vector<int16_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::UINT16:
		{
			uint16_t* iptr = (uint16_t*) &in[0];
			success = convert(out, otype, std::vector<uint16_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::INT32:
		{
			int32_t* iptr = (int32_t*) &in[0];
			success = convert(out, otype, std::vector<int32_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::UINT32:
		{
			uint32_t* iptr = (uint32_t*) &in[0];
			success = convert(out, otype, std::vector<uint32_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::INT64:
		{
			int64_t* iptr = (int64_t*) &in[0];
			success = convert(out, otype, std::vector<int64_t>(iptr, iptr + n));
		}
		break;
		case DTYPE::UINT64:
		{
			uint64_t* iptr = (uint64_t*) &in[0];
			success = convert(out, otype, std::vector<uint64_t>(iptr, iptr + n));
		}
		break;
		default:
			success = false; // unsupported type
	}
	return success;
}

}

#endif
