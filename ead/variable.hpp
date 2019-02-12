///
/// variable.hpp
/// ead
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "ead/constant.hpp"

#ifndef EAD_VARIABLE_HPP
#define EAD_VARIABLE_HPP

namespace ead
{

/// Leaf node containing data
template <typename T>
struct Variable final : public Constant<T>
{
	static Variable* get (ade::Shape shape, std::string label = "")
	{
		return Variable<T>::get(std::vector<T>(shape.n_elems(), 0),
			shape, label);
	}

	static Variable* get (T* ptr, ade::Shape shape, std::string label = "")
	{
		return new Variable<T>(ptr, shape, label);
	}

	static Variable* get (T scalar, ade::Shape shape, std::string label = "")
	{
		if (label.empty())
		{
			label = fmts::to_string(scalar);
		}
		return Variable<T>::get(std::vector<T>(shape.n_elems(),scalar),
			shape, label);
	}

	static Variable* get (std::vector<T> data, ade::Shape shape,
		std::string label = "")
	{
		if (data.size() != shape.n_elems())
		{
			logs::fatalf("cannot create variable with data size %d "
				"against shape %s", data.size(), shape.to_string().c_str());
		}
		return new Variable<T>(data.data(), shape, label);
	}

	static Variable* get (const Variable& other)
	{
		return new Variable<T>(other);
	}

	static Variable* get (Variable&& other)
	{
		return new Variable<T>(std::move(other));
	}

	Variable<T>& operator = (const Variable<T>& other) = default;

	Variable<T>& operator = (Variable<T>&& other) = default;

	/// Assign vectorized data to data source
	Variable<T>& operator = (std::vector<T> input)
	{
		size_t ninput = input.size();
		if (this->shape_.n_elems() != ninput)
		{
			logs::fatalf("cannot assign vector of %d elements to "
				"internal data of shape %s", ninput,
				this->shape_.to_string().c_str());
		}
		std::memcpy(this->data_.data(), input.data(), ninput * sizeof(T));
		return *this;
	}

	Variable<T>& operator = (const TensorT<T>& input)
	{
		this->data_ = input;
		return *this;
	}

	void assign (void* input,
		age::_GENERATED_DTYPE dtype, ade::Shape shape)
	{
		this->data_ = *raw_to_tensorptr<T>(input, dtype, shape);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_ + "(" + this->shape_.to_string() + ")";
	}

	bool is_const (void) const override
	{
		return false;
	}

	/// Label for distinguishing variable nodes
	std::string label_;

private:
	Variable (T* data, ade::Shape shape, std::string label) :
		Constant<T>(data, shape), label_(label) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

};

/// Smart pointer for variable nodes
template <typename T>
using VarptrT = std::shared_ptr<Variable<T>>;

template <typename T>
NodeptrT<T> to_node (VarptrT<T> vp)
{
	return std::make_shared<LeafNode<T>>(vp);
}

}

#endif // EAD_VARIABLE_HPP