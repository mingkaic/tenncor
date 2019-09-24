#include "tag/prop.hpp"

#include "eteq/ileaf.hpp"
#include "eteq/inode.hpp"

#ifndef ETEQ_CONSTANT_HPP
#define ETEQ_CONSTANT_HPP

namespace eteq
{

static const size_t label_limit = 5;

template <typename T>
struct Constant final : public iLeaf<T>
{
	static Constant<T>* get (T* data, teq::Shape shape);

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
		const T* data = this->data_.data();
		if (is_scalar())
		{
			if (0 == data[0]) // prevent -0
			{
				return "0";
			}
			return fmts::to_string(data[0]);
		}
		size_t nelems = this->shape_.n_elems();
		auto out = fmts::to_string(data,
			data + std::min(label_limit, nelems));
		if (nelems > label_limit)
		{
			out += "...";
		}
		return out;
	}

	/// Implementation of iLeaf
	bool is_const (void) const override
	{
		return true;
	}

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

template <typename T>
struct ConstantNode final : public iNode<T>
{
	ConstantNode (std::shared_ptr<Constant<T>> cst) : cst_(cst) {}

	T* data (void) override
	{
		return (T*) cst_->data();
	}

	void update (void) override {}

	teq::TensptrT get_tensor (void) const override
	{
		return cst_;
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

template <typename T>
NodeptrT<T> make_constant_scalar (T scalar, teq::Shape shape)
{
	auto out = std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get_scalar(scalar, shape))
	);
	tag::get_property_reg().property_tag(out->get_tensor(), tag::immutable_tag);
	return out;
}

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
