///
/// shaped_arr.hpp
/// eteq
///
/// Purpose:
/// Define a data-shape representation to feed in to the variable
///

#include "teq/shape.hpp"

#include "eteq/generated/pyapi.hpp"

#ifndef ETEQ_SHAPED_ARR_HPP
#define ETEQ_SHAPED_ARR_HPP

namespace eteq
{

/// Shaped array wraps around a vector and shape
template <typename T>
struct ShapedArr final
{
	ShapedArr (void) = default;

	ShapedArr (teq::Shape shape, T data = 0) :
		data_(shape.n_elems(), data), shape_(shape) {}

	/// Vector of size equal to shape_.n_elems()
	std::vector<T> data_;

	/// Tensor shape of data_
	teq::Shape shape_;
};

}

#endif // ETEQ_SHAPED_ARR_HPP
