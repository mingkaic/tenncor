///
/// shaped_arr.hpp
/// trainer
///
/// Purpose:
/// Define a shaped data representation
///

#ifndef TRAINER_SHAPED_ARR_HPP
#define TRAINER_SHAPED_ARR_HPP

#include "internal/teq/shape.hpp"

namespace trainer
{

/// Shaped array wraps around a vector and shape
template <typename T>
struct ShapedArr final
{
	ShapedArr (void) = default;

	ShapedArr (teq::Shape shape, T data = 0) :
		data_(shape.n_elems(), data), shape_(shape) {}

	ShapedArr (teq::Shape shape, const std::vector<T>& data) :
		data_(data), shape_(shape) {}

	/// Vector of size equal to shape_.n_elems()
	std::vector<T> data_;

	/// Tensor shape of data_
	teq::Shape shape_;
};

}

#endif // TRAINER_SHAPED_ARR_HPP
