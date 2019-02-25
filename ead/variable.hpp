///
/// variable.hpp
/// ead
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "ead/ileaf.hpp"
#include "ead/inode.hpp"

#ifndef EAD_VARIABLE_HPP
#define EAD_VARIABLE_HPP

namespace ead
{

/// Leaf node containing data
template <typename T>
struct Variable final : public iLeaf<T>
{
	static Variable<T>* get (ade::Shape shape, std::string label = "")
	{
		return Variable<T>::get(std::vector<T>(shape.n_elems(), 0),
			shape, label);
	}

	static Variable<T>* get (T* ptr, ade::Shape shape, std::string label = "")
	{
		return new Variable<T>(ptr, shape, label);
	}

	static Variable<T>* get (T scalar, ade::Shape shape, std::string label = "")
	{
		if (label.empty())
		{
			label = fmts::to_string(scalar);
		}
		return Variable<T>::get(std::vector<T>(shape.n_elems(),scalar),
			shape, label);
	}

	static Variable<T>* get (std::vector<T> data, ade::Shape shape,
		std::string label = "")
	{
		if (data.size() != shape.n_elems())
		{
			logs::fatalf("cannot create variable with data size %d "
				"against shape %s", data.size(), shape.to_string().c_str());
		}
		return new Variable<T>(data.data(), shape, label);
	}

	static Variable<T>* get (const Variable<T>& other)
	{
		return new Variable<T>(other);
	}

	static Variable<T>* get (Variable<T>&& other)
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

	void assign (void* input, age::_GENERATED_DTYPE dtype, ade::Shape shape)
	{
		std::vector<T> data;
		age::type_convert(data, input, dtype, shape.n_elems());
		this->data_ = make_tensmap<T>(data.data(), shape);;
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
		iLeaf<T>(data, shape), label_(label) {}

	Variable (const Variable<T>& other) = default;

	Variable (Variable<T>&& other) = default;

};

template <typename T>
struct VariableNode final : public iNode<T>
{
	VariableNode (std::shared_ptr<Variable<T>> var) : var_(var) {}

	T* data (void) override
	{
		return (T*) var_->data();
	}

	void update (void) override {}

	ade::TensptrT get_tensor (void) override
	{
		return var_;
	}

	void assign (T* input, ade::Shape shape)
	{
		var_->assign(input, age::get_type<T>(), shape);
	}

	void assign (TensMapT<double>* tensmap)
	{
		var_->assign(tensmap->data(), age::get_type<T>(), get_shape(*tensmap));
	}

	std::string get_label (void) const
	{
		return var_->label_;
	}

private:
	std::shared_ptr<Variable<T>> var_;
};

template <typename T>
using VarptrT = std::shared_ptr<VariableNode<T>>;

template <typename T>
NodeptrT<T> convert_to_node (VarptrT<T> var)
{
	return std::static_pointer_cast<iNode<double>>(var);
}

template <typename T>
VarptrT<T> make_variable_scalar (T scalar, ade::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(scalar, shape, label))
	);
}

template <typename T>
VarptrT<T> make_variable (ade::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(shape, label))
	);
}

template <typename T>
VarptrT<T> make_variable (T* data, ade::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(data, shape, label))
	);
}

}

#endif // EAD_VARIABLE_HPP
