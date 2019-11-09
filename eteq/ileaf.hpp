///
/// ileaf.hpp
/// eteq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "teq/ileaf.hpp"

#include "eigen/generated/dtype.hpp"
#include "eigen/eigen.hpp"

#ifndef ETEQ_ILEAF_HPP
#define ETEQ_ILEAF_HPP

namespace eteq
{

/// iLeaf extension of TEQ iLeaf containing Eigen data objects
template <typename T>
struct iLeaf : public teq::iLeaf
{
	virtual ~iLeaf (void) = default;

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iData
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return sizeof(T) * shape_.n_elems();
	}

protected:
	iLeaf (T* data, teq::Shape shape) :
		data_(eigen::make_tensmap(data, shape)),
		shape_(shape) {}

	/// Data Source
	eigen::TensorT<T> data_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;
};

}

#endif // ETEQ_ILEAF_HPP
