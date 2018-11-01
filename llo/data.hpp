///
/// data.hpp
/// llo
///
/// Purpose:
/// Define tensor data to pass up the equation graph
///

#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

/// GenericData for holding data when passing up the tensor graph
struct GenericData
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, DTYPE dtype);

	/// Return data converted to specified input type
	GenericData convert_to (DTYPE dtype) const;

	/// Smartpointer to a block of untyped data
	std::shared_ptr<char> data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Data type of data_
	DTYPE dtype_ = BAD;
};

/// GenericRef for holding data
/// Ref uses raw pointer instead of shared, so it's memory unsafe
struct GenericRef
{
	GenericRef (char* data, ade::Shape shape, DTYPE dtype) :
		data_(data), shape_(shape), dtype_(dtype) {}

	GenericRef (GenericData& generic) :
		data_(generic.data_.get()),
		shape_(generic.shape_), dtype_(generic.dtype_) {}

	/// Raw pointer to a block of untyped data
	char* data_;

	/// Shape of data_
	ade::Shape shape_;

	/// Data type of data_
	DTYPE dtype_;
};

void fill_one (char* cptr, size_t n, DTYPE dtype);

}

#endif // LLO_DATA_HPP
