//
//  packer.cpp
//  lead
//

#include "lead/include/packer.hpp"

#include "clay/memory.hpp"
#include "clay/error.hpp"

#ifdef LEAD_PACKER_HPP

namespace lead
{

std::shared_ptr<char> unpack_data (const google::protobuf::Any& src,
	clay::DTYPE dtype)
{
	std::shared_ptr<char> out;
	size_t nout;
	switch (dtype)
	{
		case tenncor::DOUBLE:
		{
			tenncor::DoubleArr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<double>& vec = arr.data();
			nout = vec.size();
			size_t nb = nout * sizeof(double);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		case tenncor::FLOAT:
		{
			tenncor::FloatArr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<float>& vec = arr.data();
			nout = vec.size();
			size_t nb = vec.size() * sizeof(float);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		case tenncor::INT8:
		case tenncor::UINT8:
		{
			tenncor::ByteArr arr;
			src.UnpackTo(&arr);
			std::string vec = arr.data();
			nout = vec.size();
			out = clay::make_char(nout);
			std::memcpy(out.get(), vec.c_str(), nout);
		}
		break;
		case tenncor::INT16:
		{
			tenncor::Int32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& vec = arr.data();
			std::vector<int16_t> conv(vec.begin(), vec.end());
			nout = vec.size();
			size_t nb = nout * sizeof(uint16_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &conv[0], nb);
		}
		break;
		case tenncor::UINT16:
		{
			tenncor::Uint32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& vec = arr.data();
			std::vector<uint16_t> conv(vec.begin(), vec.end());
			nout = vec.size();
			size_t nb = nout * sizeof(uint16_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &conv[0], nb);
		}
		break;
		case tenncor::INT32:
		{
			tenncor::Int32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& vec = arr.data();
			nout = vec.size();
			size_t nb = nout * sizeof(int32_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		case tenncor::UINT32:
		{
			tenncor::Uint32Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& vec = arr.data();
			nout = vec.size();
			size_t nb = nout * sizeof(uint32_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		case tenncor::INT64:
		{
			tenncor::Int64Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<int64_t>& vec = arr.data();
			nout = vec.size();
			size_t nb = nout * sizeof(int64_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		case tenncor::UINT64:
		{
			tenncor::Uint64Arr arr;
			src.UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint64_t>& vec = arr.data();
			nout = vec.size();
			size_t nb = nout * sizeof(uint64_t);
			out = clay::make_char(nb);
			std::memcpy(out.get(), &vec[0], nb);
		}
		break;
		default:
			throw clay::UnsupportedTypeError((clay::DTYPE) dtype);
	}
	return out;
}

void pack_data (google::protobuf::Any* dest, std::shared_ptr<char> src,
	size_t n, clay::DTYPE dtype)
{
	switch (dtype)
	{
		case tenncor::DOUBLE:
		{
			tenncor::DoubleArr arr;
			double* dptr = (double*) src.get();
			std::vector<double> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<double> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::FLOAT:
		{
			tenncor::FloatArr arr;
			float* dptr = (float*) src.get();
			std::vector<float> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<float> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::INT8:
		case tenncor::UINT8:
		{
			tenncor::ByteArr arr;
			arr.set_data(src.get(), n);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::INT16:
		{
			tenncor::Int32Arr arr;
			int16_t* dptr = (int16_t*) src.get();
			std::vector<int16_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<int32_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::UINT16:
		{
			tenncor::Uint32Arr arr;
			uint16_t* dptr = (uint16_t*) src.get();
			std::vector<uint16_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<uint32_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::INT32:
		{
			tenncor::Int32Arr arr;
			int32_t* dptr = (int32_t*) src.get();
			std::vector<int32_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<int32_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::UINT32:
		{
			tenncor::Uint32Arr arr;
			uint32_t* dptr = (uint32_t*) src.get();
			std::vector<uint32_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<uint32_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::INT64:
		{
			tenncor::Int64Arr arr;
			int64_t* dptr = (int64_t*) src.get();
			std::vector<int64_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<int64_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		case tenncor::UINT64:
		{
			tenncor::Uint64Arr arr;
			uint64_t* dptr = (uint64_t*) src.get();
			std::vector<uint64_t> vec(dptr, dptr + n);
			google::protobuf::RepeatedField<uint64_t> field(vec.begin(), vec.end());
			arr.mutable_data()->Swap(&field);
			dest->PackFrom(arr);
		}
		break;
		default:
			throw clay::UnsupportedTypeError((clay::DTYPE) dtype);
	}
}

}

#endif
