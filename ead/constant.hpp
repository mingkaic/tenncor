#include "ead/ileaf.hpp"
#include "ead/inode.hpp"

#ifndef EAD_CONSTANT_HPP
#define EAD_CONSTANT_HPP

namespace ead
{

template <typename T>
struct Constant final : public iLeaf<T>
{
	static Constant* get (T* data, ade::Shape shape)
	{
		return new Constant(data, shape);
	}

	static Constant* get (T scalar, ade::Shape shape)
	{
		size_t n = shape.n_elems();
		T buffer[n];
		std::fill(buffer, buffer + n, scalar);
		return new Constant(buffer, shape);
	}

	Constant (const Constant<T>& other) = delete;

	Constant (Constant<T>&& other) = delete;

	Constant<T>& operator = (const Constant<T>& other) = delete;

	Constant<T>& operator = (Constant<T>&& other) = delete;

	bool is_const (void) const override
	{
		return true;
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
NodeptrT<T> make_constant_scalar (T scalar, ade::Shape shape)
{
	return std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(scalar, shape))
	);
}

template <typename T>
NodeptrT<T> make_constant (T* data, ade::Shape shape)
{
	return std::make_shared<ConstantNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(data, shape))
	);
}

}

#endif // EAD_CONSTANT_HPP
