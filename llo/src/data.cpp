#include "util/error.hpp"

#include "llo/data.hpp"

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
		default:
			util::handle_error("invalid output type",
				util::ErrArg<std::string>("output.type", name_type(out_type)));
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
		default:
			util::handle_error("invalid input type",
				util::ErrArg<std::string>("input.type", name_type(dtype_)));
	}
	return out;
}

#undef CONVERT
