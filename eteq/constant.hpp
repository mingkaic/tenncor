///
/// constant.hpp
/// eteq
///
/// Purpose:
/// Implement constant leaf tensor
///

#include "tag/prop.hpp"

#include "eteq/ileaf.hpp"
#include "eteq/inode.hpp"

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
	static Constant<T>* get (T* data, teq::Shape shape);

	/// Return Constant tensor containing scalar expanded to fill shape
	static Constant<T>* get_scalar (T scalar, teq::Shape shape)
	{
		size_t n = shape.n_elems();
		T buffer[n];
		std::fill(buffer, buffer + n, scalar);
		return Constant<T>::get(buffer, shape);
	}

	Constant (const Constant<T>& other) = delete;

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
};

/// Constant's node wrapper
template <typename T>
struct ConstantNode final : public iNode<T>
{
	ConstantNode (std::shared_ptr<Constant<T>> cst) : cst_(cst) {}

	/// Return deep copy of this instance (with a copied constant)
	ConstantNode<T>* clone (void) const
	{
		return static_cast<ConstantNode<T>*>(clone_impl());
	}

	/// Implementation of iNode<T>
	T* data (void) override
	{
		return (T*) cst_->data();
	}

	/// Implementation of iNode<T>
	void update (void) override {}

	/// Implementation of iNode<T>
	teq::TensptrT get_tensor (void) const override
	{
		return cst_;
	}

protected:
	iNode<T>* clone_impl (void) const override
	{
		teq::Shape shape = cst_->shape();
		const T* d = (const T*) cst_->data();
		std::vector<T> cpy(d, d + shape.n_elems());
		return new ConstantNode(std::shared_ptr<Constant<T>>(
			Constant<T>::get(cpy.data(), shape)));
	}

private:
	std::shared_ptr<Constant<T>> cst_;
};

template <typename T>
Constant<T>* Constant<T>::get (T* data, teq::Shape shape)
{
	static bool registered = register_builder<Constant<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<ConstantNode<T>>(
				std::static_pointer_cast<Constant<T>>(tens));
		});
	assert(registered);

	return new Constant(data, shape);
}

/// Return constant node given scalar and shape
template <typename T>
NodeptrT<T> make_constant_scalar (T scalar, teq::Shape shape)
{
	auto out = std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get_scalar(scalar, shape))
	);
	tag::get_property_reg().property_tag(out->get_tensor(), tag::immutable_tag);
	return out;
}

/// Return constant node given raw array and shape
template <typename T>
NodeptrT<T> make_constant (T* data, teq::Shape shape)
{
	auto out = std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(data, shape))
	);
	tag::get_property_reg().property_tag(out->get_tensor(), tag::immutable_tag);
	return out;
}

}

#endif // ETEQ_CONSTANT_HPP
