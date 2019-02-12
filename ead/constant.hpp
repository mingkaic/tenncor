#include "ade/ileaf.hpp"

#include "ead/tensor.hpp"
#include "ead/inode.hpp"

#ifndef EAD_CONSTANT_HPP
#define EAD_CONSTANT_HPP

namespace ead
{

template <typename T>
struct Constant : public ade::iLeaf
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

	virtual ~Constant (void) = default;

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return shape().to_string();
	}

	/// Implementation of iLeaf
	void* data (void) override
	{
		return &out_;
	}

	/// Implementation of iLeaf
	const void* data (void) const override
	{
		return &out_;
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

	virtual bool is_const (void) const
	{
		return true;
	}

protected:
	Constant (T* data, ade::Shape shape) :
		data_(get_tensmap(data, shape)),
		out_(tens_to_tensmap(data_)),
		shape_(shape) {}

	TensorT<T> data_;

	TensMapT<T> out_;

	ade::Shape shape_;
};

template <typename T>
struct LeafNode final : public iNode<T>
{
	LeafNode (std::shared_ptr<Constant<T>> cst) : cst_(cst) {}

	void update (void) override {}

	TensMapT<T>* get_tensmap (void) override
	{
		return (TensMapT<T>*) cst_->data();
	}

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
	return std::make_shared<LeafNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(scalar, shape))
	);
}

template <typename T>
NodeptrT<T> make_constant (T* data, ade::Shape shape)
{
	return std::make_shared<LeafNode<T>>(
		std::shared_ptr<Constant<T>>(Constant<T>::get(data, shape))
	);
}

}

#endif // EAD_CONSTANT_HPP