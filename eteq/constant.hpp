///
/// constant.hpp
/// eteq
///
/// Purpose:
/// Implement constant leaf tensor
///

#include "eteq/ileaf.hpp"

#ifndef ETEQ_CONSTANT_HPP
#define ETEQ_CONSTANT_HPP

namespace eteq
{

/// Constant implementation of Eigen leaf tensor
template <typename T>
struct Constant final : public iLeaf<T>
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
	std::string to_string (void) const override
	{
		return teq::const_encode<T>(this->data_.data(), this->shape_);
	}

	/// Implementation of iLeaf
	teq::Usage get_usage (void) const override
	{
		return teq::Immutable;
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
		iLeaf<T>(data, shape) {}

	Constant (const Constant<T>& other) = default;

	teq::iTensor* clone_impl (void) const override
	{
		return new Constant<T>(*this);
	}
};

/// Return constant node given scalar and shape
template <typename T>
ETensor<T> make_constant_scalar (T scalar, teq::Shape shape);

/// Return constant node filled with scalar matching link shape
template <typename T>
ETensor<T> make_constant_like (T scalar, teq::TensptrT like);

/// Return constant node given raw array and shape
template <typename T>
ETensor<T> make_constant (T* data, teq::Shape shape);

}

#endif // ETEQ_CONSTANT_HPP
