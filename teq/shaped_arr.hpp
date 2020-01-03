///
/// shaped_arr.hpp
/// teq
///
/// Purpose:
/// Define a shaped data representation
///

#include "teq/shape.hpp"

#ifndef TEQ_SHAPED_ARR_HPP
#define TEQ_SHAPED_ARR_HPP

namespace teq
{

/// Shaped array wraps around a vector and shape
template <typename T>
struct ShapedArr final
{
	ShapedArr (void) = default;

	ShapedArr (Shape shape, T data = 0) :
		data_(shape.n_elems(), data), shape_(shape) {}

	ShapedArr (Shape shape, const std::vector<T>& data) :
		data_(data), shape_(shape) {}

	/// Vector of size equal to shape_.n_elems()
	std::vector<T> data_;

	/// Tensor shape of data_
	Shape shape_;
};

}

#endif // TEQ_SHAPED_ARR_HPP
