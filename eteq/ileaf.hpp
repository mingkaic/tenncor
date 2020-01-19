///
/// ileaf.hpp
/// eteq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "eigen/device.hpp"

#ifndef ETEQ_ILEAF_HPP
#define ETEQ_ILEAF_HPP

namespace eteq
{

/// iLeaf extension of TEQ iLeaf containing Eigen data objects
template <typename T>
struct iLeaf : public teq::iLeaf
{
	virtual ~iLeaf (void) = default;

	iLeaf<T>* clone (void) const
	{
		return static_cast<iLeaf<T>*>(this->clone_impl());
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		return ref_;
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		return ref_;
	}

	/// Implementation of iTensor
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iTensor
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iTensor
	size_t nbytes (void) const override
	{
		return sizeof(T) * shape_.n_elems();
	}

protected:
	iLeaf (T* data, teq::Shape shape) :
		ref_(data, shape), shape_(shape) {}

	/// Data Source
	eigen::SrcRef<T> ref_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;
};

}

#endif // ETEQ_ILEAF_HPP
