#include "llo/data.hpp"

#ifdef LLO_DATA_HPP

namespace llo
{

struct CDeleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

GenericData::GenericData (ade::Shape shape, DTYPE dtype) :
	data_((char*) malloc(shape.n_elems() * type_size(dtype)),
		CDeleter()), shape_(shape), dtype_(dtype) {}

#define COPYOVER(TYPE) { std::vector<TYPE> temp(indata, indata + n);\
	std::memcpy(out, &temp[0], nbytes); } break;

template <typename T>
void convert (char* out, DTYPE outtype, const T* indata, size_t n)
{
	size_t nbytes = type_size(outtype) * n;
	switch (outtype)
	{
		case DOUBLE: COPYOVER(double)
		case FLOAT: COPYOVER(float)
		case INT8: COPYOVER(int8_t)
		case INT16: COPYOVER(int16_t)
		case INT32: COPYOVER(int32_t)
		case INT64: COPYOVER(int64_t)
		case UINT8: COPYOVER(uint8_t)
		case UINT16: COPYOVER(uint16_t)
		case UINT32: COPYOVER(uint32_t)
		case UINT64: COPYOVER(uint64_t)
		default: err::fatalf("invalid output type %s",
			nametype(outtype).c_str());
	}
}

#undef COPYOVER

#define CONVERT(INTYPE)\
convert<INTYPE>(data_.get(), dtype_, (const INTYPE*) indata, n); break;

void GenericData::copyover (const char* indata, DTYPE intype)
{
	size_t n = shape_.n_elems();
	if (dtype_ == intype)
	{
		std::memcpy(data_.get(), indata, type_size(dtype_) * n);
	}
	switch (intype)
	{
		case DOUBLE: CONVERT(double)
		case FLOAT: CONVERT(float)
		case INT8: CONVERT(int8_t)
		case INT16: CONVERT(int16_t)
		case INT32: CONVERT(int32_t)
		case INT64: CONVERT(int64_t)
		case UINT8: CONVERT(uint8_t)
		case UINT16: CONVERT(uint16_t)
		case UINT32: CONVERT(uint32_t)
		case UINT64: CONVERT(uint64_t)
		default: err::fatalf("invalid input type %s",
			nametype(intype).c_str());
	}
}

#undef CONVERT

}

#endif
