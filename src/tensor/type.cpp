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

namespace type
{

std::string maxval (TENS_TYPE type)
{
	std::string output;
	switch (type)
	{
		case DOUBLE:
		{
			double data = std::numeric_limits<double>::max();
			output = std::string((char*) &data, sizeof(double));
		}
		break;
		case FLOAT:
		{
			float data = std::numeric_limits<float>::max();
			output = std::string((char*) &data, sizeof(float));
		}
		break;
		case INT8:
		{
			int8_t data = std::numeric_limits<int8_t>::max();
			output = std::string((char*) &data, sizeof(int8_t));
		}
		break;
		case UINT8:
		{
			uint8_t data = std::numeric_limits<uint8_t>::max();
			output = std::string((char*) &data, sizeof(uint8_t));
		}
		break;
		case INT16:
		{
			int16_t data = std::numeric_limits<int16_t>::max();
			output = std::string((char*) &data, sizeof(int16_t));
		}
		break;
		case UINT16:
		{
			uint16_t data = std::numeric_limits<uint16_t>::max();
			output = std::string((char*) &data, sizeof(uint16_t));
		}
		break;
		case INT32:
		{
			int32_t data = std::numeric_limits<int32_t>::max();
			output = std::string((char*) &data, sizeof(int32_t));
		}
		break;
		case UINT32:
		{
			uint32_t data = std::numeric_limits<uint32_t>::max();
			output = std::string((char*) &data, sizeof(uint32_t));
		}
		break;
		case INT64:
		{
			int64_t data = std::numeric_limits<int64_t>::max();
			output = std::string((char*) &data, sizeof(int64_t));
		}
		break;
		case UINT64:
		{
			uint64_t data = std::numeric_limits<uint64_t>::max();
			output = std::string((char*) &data, sizeof(uint64_t));
		}
		break;
		default:
			throw nnutils::unsupported_type_error(type);
	}
	return output;
}

std::string minval (TENS_TYPE type)
{
	std::string output;
	switch (type)
	{
		case DOUBLE:
		{
			double data = std::numeric_limits<double>::min();
			output = std::string((char*) &data, sizeof(double));
		}
		break;
		case FLOAT:
		{
			float data = std::numeric_limits<float>::min();
			output = std::string((char*) &data, sizeof(float));
		}
		break;
		case INT8:
		{
			int8_t data = std::numeric_limits<int8_t>::min();
			output = std::string((char*) &data, sizeof(int8_t));
		}
		break;
		case UINT8:
		{
			uint8_t data = std::numeric_limits<uint8_t>::min();
			output = std::string((char*) &data, sizeof(uint8_t));
		}
		break;
		case INT16:
		{
			int16_t data = std::numeric_limits<int16_t>::min();
			output = std::string((char*) &data, sizeof(int16_t));
		}
		break;
		case UINT16:
		{
			uint16_t data = std::numeric_limits<uint16_t>::min();
			output = std::string((char*) &data, sizeof(uint16_t));
		}
		break;
		case INT32:
		{
			int32_t data = std::numeric_limits<int32_t>::min();
			output = std::string((char*) &data, sizeof(int32_t));
		}
		break;
		case UINT32:
		{
			uint32_t data = std::numeric_limits<uint32_t>::min();
			output = std::string((char*) &data, sizeof(uint32_t));
		}
		break;
		case INT64:
		{
			int64_t data = std::numeric_limits<int64_t>::min();
			output = std::string((char*) &data, sizeof(int64_t));
		}
		break;
		case UINT64:
		{
			uint64_t data = std::numeric_limits<uint64_t>::min();
			output = std::string((char*) &data, sizeof(uint64_t));
		}
		break;
		default:
			throw nnutils::unsupported_type_error(type);
	}
	return output;
}

std::string zeroval (TENS_TYPE type)
{
	return std::string(type_size(type), 0);
}

}

}

#endif
