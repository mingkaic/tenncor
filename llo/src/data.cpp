#include <cstring>

#include "log/log.hpp"

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
	data_((char*) malloc(shape.n_elems() * type_size(dtype)), CDeleter()),
	shape_(shape), dtype_(dtype) {}

#define COPYOVER(TYPE) { std::vector<TYPE> temp(data, data + n);\
	std::memcpy(out, &temp[0], nbytes); } break;

template <typename T>
static void to_generic (char* out, DTYPE out_type, T* data, size_t n)
{
	size_t nbytes = n * type_size(out_type);
	switch (out_type)
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
		default: ade::fatalf("invalid output type %s",
			nametype(out_type).c_str());
	}
}

#undef COPYOVER

#define CONVERT(TYPE)\
to_generic<TYPE>(out.data_.get(), out_type,\
(TYPE*) data_.get(), shape_.n_elems()); break;

GenericData GenericData::convert_to (DTYPE out_type) const
{
	if (out_type == dtype_)
	{
		return *this;
	}
	GenericData out(shape_, out_type);
	switch (dtype_)
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
		default: ade::fatalf("invalid input type %s",
			nametype(dtype_).c_str());
	}
	return out;
}

#undef CONVERT

#define FILL_ONE(TYPE){ TYPE* ptr = (TYPE*) cptr;\
std::fill(ptr, ptr + n, (TYPE) 1); } break;

// fill all elements of specified type under cptr with values of 1
void fill_one (char* cptr, size_t n, DTYPE dtype)
{
	switch (dtype)
	{
		case DOUBLE: FILL_ONE(double)
		case FLOAT: FILL_ONE(float)
		case INT8: FILL_ONE(int8_t)
		case INT16: FILL_ONE(int16_t)
		case INT32: FILL_ONE(int32_t)
		case INT64: FILL_ONE(int64_t)
		case UINT8: FILL_ONE(uint8_t)
		case UINT16: FILL_ONE(uint16_t)
		case UINT32: FILL_ONE(uint32_t)
		case UINT64: FILL_ONE(uint64_t)
		default: ade::fatal("filling unknown type");
	}
}

#undef FILL_ONE

}

#endif
