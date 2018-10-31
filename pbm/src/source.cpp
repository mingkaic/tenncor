#include "ade/log/log.hpp"

#include "pbm/source.hpp"

#define PACK_DATA(TYPE)\
TYPE* ptr = (TYPE*) data.data_.get();\
google::protobuf::RepeatedField<TYPE> vec(ptr, ptr + nelems);\
arr->mutable_data()->Swap(&vec);

void save_data (tenncor::Source* out, llo::iSource* in)
{
	const ade::Shape& shape = in->inner()->shape();
	out->set_shape(std::string(shape.begin(), shape.end()));
	llo::GenericData data = in->data(in->native_type());
	size_t nelems = data.shape_.n_elems();
	switch (data.dtype_)
	{
		case llo::DOUBLE:
		{
			auto arr = out->mutable_double_arrs();
			PACK_DATA(double)
		}
		break;
		case llo::FLOAT:
		{
			auto arr = out->mutable_float_arrs();
			PACK_DATA(float)
		}
		break;
		case llo::INT8:
		{
			auto arr = out->mutable_sbyte_arrs();
			char* ptr = data.data_.get();
			arr->set_data(std::string(ptr, ptr + nelems));
		}
		break;
		case llo::UINT8:
		{
			auto arr = out->mutable_ubyte_arrs();
			char* ptr = data.data_.get();
			arr->set_data(std::string(ptr, ptr + nelems));
		}
		break;
		case llo::INT16:
		{
			auto arr = out->mutable_sshort_arrs();
			int16_t* ptr = (int16_t*) data.data_.get();
			std::vector<int16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<int32_t> vec(
				temp.begin(), temp.end());
			arr->mutable_data()->Swap(&vec);
		}
		break;
		case llo::INT32:
		{
			auto arr = out->mutable_sint_arrs();
			PACK_DATA(int32_t)
		}
		break;
		case llo::INT64:
		{
			auto arr = out->mutable_slong_arrs();
			PACK_DATA(int64_t)
		}
		break;
		case llo::UINT16:
		{
			auto arr = out->mutable_ushort_arrs();
			uint16_t* ptr = (uint16_t*) data.data_.get();
			std::vector<uint16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<uint32_t> vec(
				temp.begin(), temp.end());
			arr->mutable_data()->Swap(&vec);
		}
		break;
		case llo::UINT32:
		{
			auto arr = out->mutable_uint_arrs();
			PACK_DATA(uint32_t)
		}
		break;
		case llo::UINT64:
		{
			auto arr = out->mutable_ulong_arrs();
			PACK_DATA(uint64_t)
		}
		break;
		default:
			ade::error("cannot serialize badly typed node... skipping");
	}
}

#undef PACK_DATA

#define UNPACK_SOURCE(TYPE)\
auto vec = arr.data();\
return llo::Source<TYPE>::get(shape,\
	std::vector<TYPE>(vec.begin(), vec.end()));

llo::DataNode load_source (const tenncor::Source& source)
{
	std::string sstr = source.shape();
	ade::Shape shape(std::vector<ade::DimT>(sstr.begin(), sstr.end()));
	switch (source.data_case())
	{
		case tenncor::Source::DataCase::kDoubleArrs:
		{
			auto arr = source.double_arrs();
			UNPACK_SOURCE(double)
		}
		case tenncor::Source::DataCase::kFloatArrs:
		{
			auto arr = source.float_arrs();
			UNPACK_SOURCE(float)
		}
		case tenncor::Source::DataCase::kSbyteArrs:
		{
			auto arr = source.sbyte_arrs();
			UNPACK_SOURCE(int8_t)
		}
		case tenncor::Source::DataCase::kUbyteArrs:
		{
			auto arr = source.ubyte_arrs();
			UNPACK_SOURCE(uint8_t)
		}
		break;
		case tenncor::Source::DataCase::kSshortArrs:
		{
			auto arr = source.sshort_arrs();
			UNPACK_SOURCE(int16_t)
		}
		break;
		case tenncor::Source::DataCase::kSintArrs:
		{
			auto arr = source.sint_arrs();
			UNPACK_SOURCE(int32_t)
		}
		break;
		case tenncor::Source::DataCase::kSlongArrs:
		{
			auto arr = source.slong_arrs();
			UNPACK_SOURCE(int64_t)
		}
		break;
		case tenncor::Source::DataCase::kUshortArrs:
		{
			auto arr = source.ushort_arrs();
			UNPACK_SOURCE(uint16_t)
		}
		break;
		case tenncor::Source::DataCase::kUintArrs:
		{
			auto arr = source.uint_arrs();
			UNPACK_SOURCE(uint32_t)
		}
		break;
		case tenncor::Source::DataCase::kUlongArrs:
		{
			auto arr = source.ulong_arrs();
			UNPACK_SOURCE(uint64_t)
		}
		break;
		default:
			ade::fatal("cannot load source"); // todo: make more informative
	}
}

#undef UNPACK_SOURCE
