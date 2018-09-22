#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

struct GenericData
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, DTYPE dtype);

	GenericData convert_to (DTYPE dtype) const;

	std::shared_ptr<char> data_;

	ade::Shape shape_;
	DTYPE dtype_ = BAD;
};

struct GenericRef
{
	GenericRef (char* data, ade::Shape shape, DTYPE dtype) :
		data_(data), shape_(shape), dtype_(dtype) {}

	GenericRef (GenericData& generic) :
		data_(generic.data_.get()),
		shape_(generic.shape_), dtype_(generic.dtype_) {}

	char* data_;
	ade::Shape shape_;
	DTYPE dtype_;
};

}

#endif /* LLO_DATA_HPP */
