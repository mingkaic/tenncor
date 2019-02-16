///
/// ileaf.hpp
/// ead
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "ade/ileaf.hpp"

#include "ead/tensor.hpp"

#ifndef EAD_ILEAF_HPP
#define EAD_ILEAF_HPP

namespace ead
{

template <typename T>
struct iLeaf : public ade::iLeaf
{
	virtual ~iLeaf (void) = default;

	iLeaf<T>& operator = (const iLeaf<T>& other)
	{
		if (this != &other)
		{
			// out_ map must reference new copied data
			data_ = other.data_; // Eigen supported deep copy
			out_ = tens_to_tensmap(data_);
			shape_ = shape_;
		}
		return *this;
	}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return fmts::to_string(data_.data()[0]) +
			"(" + shape().to_string() + ")";
	}

	/// Implementation of iLeaf
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iLeaf
	const void* data (void) const override
	{
		return data_.data();
	}

	/// Implementation of iLeaf
	size_t type_code (void) const override
	{
		return age::get_type<T>();
	}

	/// Return number of bytes in data source
	size_t nbytes (void) const
	{
		return sizeof(T) * shape_.n_elems();
	}

	TensMapT<T>* get_tensmap (void)
	{
		return &out_;
	}

	virtual bool is_const (void) const = 0;

protected:
	iLeaf (T* data, ade::Shape shape) :
		data_(ead::get_tensmap(data, shape)),
		out_(tens_to_tensmap(data_)),
		shape_(shape) {}

	iLeaf (const iLeaf<T>& other) :
		data_(other.data_),
		out_(tens_to_tensmap(data_)),
		shape_(other.shape_) {}

	/// Data Source
	TensorT<T> data_;

	// todo: get rid of this somehow
	/// TensorMap is here for functor's iEigen to reference
	TensMapT<T> out_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	ade::Shape shape_;
};

}

#endif // EAD_ILEAF_HPP
