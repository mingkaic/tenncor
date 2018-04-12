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

std::string type_convert (void* ptr, size_t n, TENS_TYPE otype, TENS_TYPE itype)
{
	assert(otype != BAD_T && itype != BAD_T);
	size_t outbytes = n * type_size(otype);
	if (otype == itype)
	{
		return std::string((char*) ptr, outbytes);
	}
	switch (otype)
	{
		case DOUBLE:
			return std::string((char*) &type_convert<double>(ptr, n, itype)[0], outbytes);
		case FLOAT:
			return std::string((char*) &type_convert<float>(ptr, n, itype)[0], outbytes);
		case INT8:
			return std::string((char*) &type_convert<int8_t>(ptr, n, itype)[0], outbytes);
		case UINT8:
			return std::string((char*) &type_convert<uint8_t>(ptr, n, itype)[0], outbytes);
		case INT16:
			return std::string((char*) &type_convert<int16_t>(ptr, n, itype)[0], outbytes);
		case UINT16:
			return std::string((char*) &type_convert<uint16_t>(ptr, n, itype)[0], outbytes);
		case INT32:
			return std::string((char*) &type_convert<int32_t>(ptr, n, itype)[0], outbytes);
		case UINT32:
			return std::string((char*) &type_convert<uint32_t>(ptr, n, itype)[0], outbytes);
		case INT64:
			return std::string((char*) &type_convert<int64_t>(ptr, n, itype)[0], outbytes);
		case UINT64:
			return std::string((char*) &type_convert<uint64_t>(ptr, n, itype)[0], outbytes);
		default:
			throw nnutils::unsupported_type_error(itype);
	}
}

void serialize_data (google::protobuf::Any* dest, 
	void* ptr, TENS_TYPE type, size_t n)
{
	switch (type)
	{
		case DOUBLE:
		{
			tenncor::DoubleArr arr;
			double* dptr = (double*) ptr;
			std::vector<double> data(dptr, dptr + n);
			google::protobuf::RepeatedField<double> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case FLOAT:
		{
			tenncor::FloatArr arr;
			float* dptr = (float*) ptr;
			std::vector<float> data(dptr, dptr + n);
			google::protobuf::RepeatedField<float> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case INT8:
		case UINT8:
		{
			tenncor::ByteArr arr;
			arr.set_data(ptr, n);
			dest->PackFrom(arr);
		}
		break;
		case INT16:
		{
			tenncor::Int32Arr arr;
			int16_t* dptr = (int16_t*) ptr;
			std::vector<int16_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<int32_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case UINT16:
		{
			tenncor::Uint32Arr arr;
			uint16_t* dptr = (uint16_t*) ptr;
			std::vector<uint16_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<uint32_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case INT32:
		{
			tenncor::Int32Arr arr;
			int32_t* dptr = (int32_t*) ptr;
			std::vector<int32_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<int32_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case UINT32:
		{
			tenncor::Uint32Arr arr;
			uint32_t* dptr = (uint32_t*) ptr;
			std::vector<uint32_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<uint32_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case INT64:
		{
			tenncor::Int64Arr arr;
			int64_t* dptr = (int64_t*) ptr;
			std::vector<int64_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<int64_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case UINT64:
		{
			tenncor::Uint64Arr arr;
			uint64_t* dptr = (uint64_t*) ptr;
			std::vector<uint64_t> data(dptr, dptr + n);
			google::protobuf::RepeatedField<uint64_t> field(data.begin(), data.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		default:
			throw nnutils::unsupported_type_error(type);
	}
}

std::shared_ptr<void> deserialize_data (const google::protobuf::Any& src,
	TENS_TYPE type, size_t* n)
{
	std::shared_ptr<void> out;
	size_t nout;
	switch (type)
	{
		case DOUBLE:
		{
			tenncor::DoubleArr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<double>& data = arr.data();
			nout = data.size();
			size_t nb = nout * sizeof(double);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		case FLOAT:
		{
			tenncor::FloatArr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<float>& data = arr.data();
			nout = data.size();
			size_t nb = data.size() * sizeof(float);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		case INT8:
		case UINT8:
		{
			tenncor::ByteArr arr;
			src.UnpackTo(&arr);
			std::string data = arr.data();
			nout = data.size();
			out = nnutils::make_svoid(nout);
			std::memcpy(out.get(), data.c_str(), nout);
		}
		break;
		case INT16:
		{
			tenncor::Int32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& data = arr.data();
			std::vector<int16_t> conv(data.begin(), data.end());
			nout = data.size();
			size_t nb = nout * sizeof(uint16_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &conv[0], nb);
		}
		break;
		case UINT16:
		{
			tenncor::Uint32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& data = arr.data();
			std::vector<uint16_t> conv(data.begin(), data.end());
			nout = data.size();
			size_t nb = nout * sizeof(uint16_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &conv[0], nb);
		}
		break;
		case INT32:
		{
			tenncor::Int32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& data = arr.data();
			nout = data.size();
			size_t nb = nout * sizeof(int32_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		case UINT32:
		{
			tenncor::Uint32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& data = arr.data();
			nout = data.size();
			size_t nb = nout * sizeof(uint32_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		case INT64:
		{
			tenncor::Int64Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int64_t>& data = arr.data();
			nout = data.size();
			size_t nb = nout * sizeof(int64_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		case UINT64:
		{
			tenncor::Uint64Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint64_t>& data = arr.data();
			nout = data.size();
			size_t nb = nout * sizeof(uint64_t);
			out = nnutils::make_svoid(nb);
			std::memcpy(out.get(), &data[0], nb);
		}
		break;
		default:
			throw nnutils::unsupported_type_error(type);
	}
	if (n != nullptr)
	{
		*n = nout;
	}
	return out;
}

}

#endif
