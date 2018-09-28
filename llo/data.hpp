/*!
 *
 *  data.hpp
 *  llo
 *
 *  Purpose:
 *  define typed data for evaluating the operation tree
 *
 */

#include "ade/tensor.hpp"

#include "llo/dtype.hpp"

#ifndef LLO_DATA_HPP
#define LLO_DATA_HPP

namespace llo
{

/*! GenericData for encapsulating data to up the tensor tree */
struct GenericData
{
	GenericData (void) = default;

	GenericData (ade::Shape shape, DTYPE dtype);

	/*! convert data to specified input type */
	GenericData convert_to (DTYPE dtype) const;

	/*! smartpointer to a block of generic data */
	std::shared_ptr<char> data_;

	/*! shape data to hold size info */
	ade::Shape shape_;
	/*! type of owned data */
	DTYPE dtype_ = BAD;
};

/*! GenericRef encapsulating data through raw pointer (memory unsafe) */
struct GenericRef
{
	GenericRef (char* data, ade::Shape shape, DTYPE dtype) :
		data_(data), shape_(shape), dtype_(dtype) {}

	GenericRef (GenericData& generic) :
		data_(generic.data_.get()),
		shape_(generic.shape_), dtype_(generic.dtype_) {}

	/*! raw pointer to a block of generic data */
	char* data_;
	/*! shape data to hold size info */
	ade::Shape shape_;
	/*! type of referenced data */
	DTYPE dtype_;
};

}

#endif /* LLO_DATA_HPP */
