///
/// variable.hpp
/// eteq
///
/// Purpose:
/// Define data structures for owning, and passing
///	generalized and type-specific data
///

#include "eteq/ileaf.hpp"
#include "eteq/inode.hpp"
#include "eteq/shaped_arr.hpp"

#ifndef ETEQ_VARIABLE_HPP
#define ETEQ_VARIABLE_HPP

namespace eteq
{

/// Leaf node containing data
template <typename T>
struct Variable final : public iLeaf<T>
{
	static Variable<T>* get (T* ptr, teq::Shape shape, std::string label = "");

	static Variable<T>* get (teq::Shape shape, std::string label = "")
	{
		return Variable<T>::get(std::vector<T>(shape.n_elems(), 0),
			shape, label);
	}

	static Variable<T>* get (T scalar, teq::Shape shape, std::string label = "")
	{
		if (label.empty())
		{
			label = fmts::to_string(scalar);
		}
		return Variable<T>::get(std::vector<T>(shape.n_elems(),scalar),
			shape, label);
	}

	static Variable<T>* get (std::vector<T> data, teq::Shape shape,
		std::string label = "")
	{
		if (data.size() != shape.n_elems())
		{
			logs::fatalf("cannot create variable with data size %d "
				"against shape %s", data.size(), shape.to_string().c_str());
		}
		return Variable<T>::get(data.data(), shape, label);
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

	void assign (const void* input, egen::_GENERATED_DTYPE dtype, teq::Shape shape)
	{
		if (false == shape.compatible_after(this->shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), this->shape_.to_string().c_str());
		}
		std::vector<T> data;
		egen::type_convert(data, input, dtype, shape.n_elems());
		this->data_ = make_tensmap<T>(data.data(), shape);
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return label_;
	}

	/// Implementation of iLeaf
	bool is_const (void) const override
	{
		return false;
	}

	/// Label for distinguishing variable nodes
	std::string label_; // todo: make private

private:
	Variable (T* data, teq::Shape shape, std::string label) :
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

	teq::TensptrT get_tensor (void) const override
	{
		return var_;
	}

	void assign (const T* input, teq::Shape shape)
	{
		var_->assign(input, egen::get_type<T>(), shape);
	}

	void assign (const TensMapT<T>* tensmap)
	{
		var_->assign(tensmap->data(), egen::get_type<T>(), get_shape(*tensmap));
	}

	void assign (const ShapedArr<T>& arr)
	{
		var_->assign(arr.data_.data(), egen::get_type<T>(), arr.shape_);
	}

	std::string get_label (void) const
	{
		return var_->to_string();
	}

private:
	std::shared_ptr<Variable<T>> var_;
};

template <typename T>
Variable<T>* Variable<T>::get (T* ptr, teq::Shape shape, std::string label)
{
	static bool registered = register_builder<Variable<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<VariableNode<T>>(
				std::static_pointer_cast<Variable<T>>(tens));
		});
	assert(registered);

	return new Variable<T>(ptr, shape, label);
}

template <typename T>
using VarptrT = std::shared_ptr<VariableNode<T>>;

template <typename T>
NodeptrT<T> convert_to_node (VarptrT<T> var)
{
	return std::static_pointer_cast<iNode<T>>(var);
}

template <typename T>
VarptrT<T> make_variable_scalar (T scalar, teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(scalar, shape, label))
	);
}

template <typename T>
VarptrT<T> make_variable (teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(shape, label))
	);
}

template <typename T>
VarptrT<T> make_variable (T* data, teq::Shape shape, std::string label = "")
{
	return std::make_shared<VariableNode<T>>(
		std::shared_ptr<Variable<T>>(Variable<T>::get(data, shape, label))
	);
}

}

#endif // ETEQ_VARIABLE_HPP
