///
/// constant.hpp
/// eteq
///
/// Purpose:
/// Implement constant leaf tensor
///

#ifndef ETEQ_CONSTANT_HPP
#define ETEQ_CONSTANT_HPP

#include "tenncor/eteq/etens.hpp"

namespace eteq
{

/// Constant implementation of Eigen leaf tensor
template <typename T>
struct Constant final : public teq::iLeaf
{
	/// Return Constant tensor containing first
	/// shape.n_elems() values of data pointer
	static Constant<T>* get (T* data, teq::Shape shape)
	{
		return new Constant(data, shape);
	}

	Constant<T>* clone (void) const
	{
		return static_cast<Constant<T>*>(clone_impl());
	}

	Constant (Constant<T>&& other) = delete;

	Constant<T>& operator = (const Constant<T>& other) = delete;

	Constant<T>& operator = (Constant<T>&& other) = delete;

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
	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return teq::const_encode<T>((T*) this->device().data(), this->shape_);
	}

	/// Implementation of iLeaf
	teq::Usage get_usage (void) const override
	{
		return teq::IMMUTABLE;
	}

	/// Return true if constant data values are all the same, otherwise false
	bool is_scalar (void) const
	{
		const T* data = this->data_.data();
		size_t nelems = this->shape_.n_elems();
		return std::all_of(data + 1, data + nelems,
			[&](const T& e) { return e == data[0]; });
	}

private:
	Constant (T* data, teq::Shape shape) :
		ref_(data, shape), shape_(shape) {}

	Constant (const Constant<T>& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Constant<T>(*this);
	}

	/// Data Source
	eigen::SrcRef<T> ref_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;

	/// Variable metadata
	eigen::EMetadata<T> meta_ = eigen::EMetadata<T>(1);
};

/// Return constant node given scalar and shape
template <typename T>
ETensor<T> make_constant_scalar (T scalar, teq::Shape shape,
	const global::CfgMapptrT& ctx = global::context());

/// Return constant node filled with scalar matching link shape
template <typename T>
ETensor<T> make_constant_like (T scalar, teq::TensptrT like,
	const global::CfgMapptrT& ctx = global::context());

/// Return constant node given raw array and shape
template <typename T>
ETensor<T> make_constant (T* data, teq::Shape shape,
	const global::CfgMapptrT& ctx = global::context());

}

#endif // ETEQ_CONSTANT_HPP
