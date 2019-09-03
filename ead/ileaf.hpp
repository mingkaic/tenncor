///
/// ileaf.hpp
/// ead
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "ade/ileaf.hpp"

#include "ead/eigen.hpp"

#ifndef EAD_ILEAF_HPP
#define EAD_ILEAF_HPP

namespace ead
{

template <typename T>
struct iLeaf : public ade::iLeaf
{
	virtual ~iLeaf (void) = default;

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
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
		return age::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return age::name_type(age::get_type<T>());
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return sizeof(T) * shape_.n_elems();
	}

protected:
	iLeaf (T* data, ade::Shape shape) :
		data_(make_tensmap(data, shape)),
		shape_(shape) {}

	/// Data Source
	TensorT<T> data_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	ade::Shape shape_;
};

}

#endif // EAD_ILEAF_HPP
