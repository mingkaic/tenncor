#include "tag/prop.hpp"

#include "ead/ileaf.hpp"
#include "ead/inode.hpp"

#ifndef EAD_CONSTANT_HPP
#define EAD_CONSTANT_HPP

namespace ead
{

static const size_t label_limit = 5;

template <typename T>
struct Constant final : public iLeaf<T>
{
	static Constant<T>* get (T* data, ade::Shape shape);

	static Constant<T>* get_scalar (T scalar, ade::Shape shape)
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
	Constant (T* data, ade::Shape shape) :
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

	ade::TensptrT get_tensor (void) override
	{
		return cst_;
	}

private:
	std::shared_ptr<Constant<T>> cst_;
};

template <typename T>
Constant<T>* Constant<T>::get (T* data, ade::Shape shape)
{
	static bool registered = register_builder<Constant<T>,T>(
		[](ade::TensptrT tens)
		{
			return std::make_shared<ConstantNode<T>>(
				std::static_pointer_cast<Constant<T>>(tens));
		});
	assert(registered);

	return new Constant(data, shape);
}

template <typename T>
NodeptrT<T> make_constant_scalar (T scalar, ade::Shape shape)
{
	auto out = std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get_scalar(scalar, shape))
	);
	tag::property_tag(out->get_tensor(), tag::immutable_tag);
	return out;
}

template <typename T>
NodeptrT<T> make_constant (T* data, ade::Shape shape)
{
	auto out = std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(data, shape))
	);
	tag::property_tag(out->get_tensor(), tag::immutable_tag);
	return out;
}

}

#endif // EAD_CONSTANT_HPP
