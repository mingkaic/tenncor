///
/// constant.hpp
/// eteq
///
/// Purpose:
/// Implement constant leaf tensor
///

#include "tag/prop.hpp"

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

	/// Return Constant tensor containing scalar expanded to fill shape
	static Constant<T>* get_scalar (T scalar, teq::Shape shape)
	{
		size_t n = shape.n_elems();
		T buffer[n];
		std::fill(buffer, buffer + n, scalar);
		return Constant<T>::get(buffer, shape);
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
	bool is_const (void) const override
	{
		return true;
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
LinkptrT<T> make_constant_scalar (T scalar, teq::Shape shape)
{
	teq::TensptrT out(Constant<T>::get_scalar(scalar, shape));
	tag::get_property_reg().property_tag(out, tag::immutable_tag);
	return leaf_link<T>(out);
}

/// Return constant node filled with scalar matching link shape
template <typename T>
LinkptrT<T> make_constant_like (T scalar, LinkptrT<T> link)
{
	auto sign = link->shape_sign();
	if (teq::is_ambiguous(sign))
	{
		logs::fatalf("cannot create constant with ambiguous shaped %s",
			sign.to_string().c_str());
	}
	teq::TensptrT out(Constant<T>::get_scalar(scalar, teq::Shape(sign)));
	tag::get_property_reg().property_tag(out, tag::immutable_tag);
	return leaf_link<T>(out);
}

/// Return constant node given raw array and shape
template <typename T>
LinkptrT<T> make_constant (T* data, teq::Shape shape)
{
	teq::TensptrT out(Constant<T>::get(data, shape));
	tag::get_property_reg().property_tag(out, tag::immutable_tag);
	return leaf_link<T>(out);
}

}

#endif // ETEQ_CONSTANT_HPP
